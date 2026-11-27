'use strict';

const assert = require('node:assert/strict');
const {mkdtempSync, readFileSync, rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {DatabaseSync} = require('node:sqlite');
const {Readable} = require('node:stream');
const test = require('node:test');
const {gzipSync} = require('node:zlib');
const {bindKeptCurS3Reader} = require('../server/scoped_cur_s3.cjs');
const {createCurRefreshJobRunner} = require('../server/cur_refresh_job.cjs');
const {createPrivateBillingRead} = require('../server/private_billing_read.cjs');

const tenant = 'test-tenant', source = 'aws';
const manifestKey = 'fixture/export/20260101-20260201/atlas-Manifest.json';
const csv = readFileSync(join(__dirname, 'fixtures', 'cur', 'synthetic.csv'), 'utf8');
const manifestFor = hour => {
    const assemblyId = '20260202T0' + hour + '0000Z';
    return {assemblyId, reportName: 'atlas',
        billingPeriod: {start: '20260101T000000.000Z', end: '20260201T000000.000Z'},
        reportKeys: ['fixture/export/20260101-20260201/' + assemblyId + '/atlas-00001.csv.gz']};
};

function fixtureReader(h, session) {
    return bindKeptCurS3Reader({
        tenant, source, manifestKey, bucket: 'fixture-cur', expectedBucketOwner: '111122223333',
        sendGetObject: async ({Key, IfMatch}) => {
            h.cloudCalls++;
            await h.beforeRead?.({session, Key, IfMatch});
            const latest = manifestFor(h.hour);
            const etag = '"' + latest.assemblyId + '"';
            if (Key === manifestKey) {
                if (IfMatch) assert.equal(IfMatch, etag);
                return {Body: Readable.from([Buffer.from(JSON.stringify(latest))]), ETag: etag};
            }
            assert.equal(Key, latest.reportKeys[0]);
            if (h.hook) await h.hook();
            return {Body: Readable.from([gzipSync(csv.replace(',30,USD,', ',' + (25 + 5 * h.hour) + ',USD,'))])};
        },
    });
}

function setup() {
    const h = {hour: 1, opens: 0, committed: 0, cloudCalls: 0, loseAck: false, activeGeneration: 1,
        jobs: new Map(), latest: null, hook: null};
    h.seed = jobId => h.jobs.set(jobId, {tenant, jobId, source, period: '2026-01', generation: 1, mappingVersion: 0});
    h.run = createCurRefreshJobRunner({
        now: () => new Date(Date.UTC(2026, 1, 2, h.hour)),
        loadJob: async ({tenant: scope, jobId}) => scope === tenant ? h.jobs.get(jobId) : null,
        openSource: async binding => {
            h.opens++;
            assert.equal(binding.tenant, tenant);
            assert.equal(binding.source, source);
            assert.equal(binding.period, '2026-01');
            return {generation: h.activeGeneration, mappingVersion: 0, manifestKey,
                readStreamObject: fixtureReader(h, h.opens)};
        },
        commitJob: async ({tenant: scope, jobId, source: candidateSource, generation, mappingVersion, snapshot}) => {
            const job = h.jobs.get(jobId);
            assert.equal(scope, job.tenant);
            assert.equal(candidateSource, job.source);
            assert.equal(mappingVersion, job.mappingVersion);
            if (job.outcome) return job.outcome;
            if (generation !== h.activeGeneration || snapshot.assemblyId <= (h.latest?.assemblyId ?? '')) {
                return job.outcome = {status: 'not_published'};
            }
            h.committed++;
            h.latest = snapshot;
            job.outcome = {status: 'completed', publishedAt: snapshot.acceptedAt};
            if (h.loseAck) throw new Error('lost acknowledgement');
            return job.outcome;
        },
    });
    return h;
}

test('a retry checks durable job outcome before S3 and preserves a newer report', async () => {
    const h = setup();
    for (const id of ['first', 'correction', 'newer']) h.seed(id);
    assert.equal((await h.run({tenant, jobId: 'first'})).status, 'completed');
    assert.equal(h.latest.report.TotalBilledCost, 100);

    h.hour = 2;
    h.loseAck = true;
    await assert.rejects(h.run({tenant, jobId: 'correction'}), /lost acknowledgement/);
    const corrected = h.latest;
    assert.equal(corrected.report.TotalBilledCost, 105);
    const checkpoint = h.cloudCalls;
    assert.deepEqual(await h.run({tenant, jobId: 'correction'}), {status: 'completed', publishedAt: corrected.acceptedAt});
    assert.equal(h.cloudCalls, checkpoint);
    assert.deepEqual(h.latest, corrected);

    h.hour = 3;
    h.loseAck = false;
    assert.equal((await h.run({tenant, jobId: 'newer'})).status, 'completed');
    assert.equal(h.latest.report.TotalBilledCost, 110);
    assert.deepEqual(await h.run({tenant, jobId: 'correction'}), {status: 'completed', publishedAt: corrected.acceptedAt});
    assert.equal(h.latest.report.TotalBilledCost, 110);
    assert.equal(h.committed, 3);
});

test('a removed connection or another tenant cannot publish', async () => {
    const h = setup();
    h.seed('first');
    h.seed('after-disconnect');
    await assert.rejects(h.run({tenant: 'other', jobId: 'first'}), /job is unavailable/);
    assert.equal(h.opens, 0);
    await h.run({tenant, jobId: 'first'});
    const accepted = h.latest;
    h.hour = 2;
    h.hook = () => { h.activeGeneration++; };
    assert.deepEqual(await h.run({tenant, jobId: 'after-disconnect'}), {status: 'not_published'});
    assert.equal(h.committed, 1);
    assert.equal(h.latest, accepted);
});

test('expiry during final manifest verification retains the last report; retry reopens the source', async () => {
    const h = setup();
    h.seed('first');
    h.seed('correction');
    assert.equal(h.opens, 0, 'queued jobs must not acquire source access');
    const reads = [];
    h.beforeRead = ({session, Key, IfMatch}) => {
        reads.push({session, Key, IfMatch});
        if (session === 2 && IfMatch) {
            throw Object.assign(new Error('Fixture session expired.'), {name: 'ExpiredToken'});
        }
    };

    await h.run({tenant, jobId: 'first'});
    const before = structuredClone(h.latest);
    h.hour = 2;
    await assert.rejects(h.run({tenant, jobId: 'correction'}), {name: 'ExpiredToken'});
    assert.deepEqual(h.latest, before);
    assert.equal(h.committed, 1);
    assert.equal(h.jobs.get('correction').outcome, undefined);
    const failed = reads.filter(read => read.session === 2);
    assert.deepEqual(failed.map(read => read.Key), [manifestKey, manifestFor(2).reportKeys[0], manifestKey]);
    assert.equal(failed[2].IfMatch, '"' + manifestFor(2).assemblyId + '"');

    const corrected = await h.run({tenant, jobId: 'correction'});
    assert.equal(corrected.status, 'completed');
    assert.equal(h.latest.report.TotalBilledCost, 105);
    assert.notEqual(h.latest.acceptedAt, before.acceptedAt);
    assert.deepEqual(reads.filter(read => read.session === 3),
        failed.map(read => ({...read, session: 3})));
    const checkpoint = reads.length;
    assert.deepEqual(await h.run({tenant, jobId: 'correction'}), corrected);
    assert.equal(h.opens, 3);
    assert.equal(reads.length, checkpoint);
});

// A test-only host: no HTTP routes, real sessions, AWS credentials or implied
// production database choice. Independent connections model the reader/worker split.
function openTestHost(file, h) {
    const writer = new DatabaseSync(file);
    writer.exec(`
        PRAGMA synchronous = FULL;
        CREATE TABLE IF NOT EXISTS job (
            tenant TEXT, id TEXT, source TEXT, period TEXT, generation INTEGER,
            mapping INTEGER, outcome TEXT, PRIMARY KEY (tenant, id)
        );
        CREATE TABLE IF NOT EXISTS connection (
            tenant TEXT, source TEXT, generation INTEGER, mapping INTEGER,
            PRIMARY KEY (tenant, source)
        );
        CREATE TABLE IF NOT EXISTS accepted (
            tenant TEXT, source TEXT, period TEXT, assembly TEXT, revision TEXT,
            mapping INTEGER, report TEXT, accepted_at TEXT, PRIMARY KEY (tenant, source, period)
        );
    `);
    const reader = new DatabaseSync(file, {readOnly: true});
    const findJob = db => db.prepare(`
        SELECT tenant, id AS jobId, source, period, generation, mapping AS mappingVersion, outcome
        FROM job WHERE tenant = ? AND id = ?
    `);
    const getJob = findJob(writer);
    const getStatus = findJob(reader);
    const readSource = writer.prepare('SELECT generation, mapping AS mappingVersion FROM connection WHERE tenant = ? AND source = ?');
    const accept = writer.prepare(`
        INSERT INTO accepted
        SELECT tenant, source, ?, ?, ?, mapping, ?, ? FROM connection
        WHERE tenant = ? AND source = ? AND generation = ? AND mapping = ?
        ON CONFLICT (tenant, source, period) DO UPDATE SET assembly = excluded.assembly,
            revision = excluded.revision, mapping = excluded.mapping, report = excluded.report,
            accepted_at = excluded.accepted_at
        WHERE excluded.assembly > accepted.assembly
           OR (excluded.assembly = accepted.assembly AND excluded.revision = accepted.revision
               AND excluded.mapping > accepted.mapping)
    `);
    const getReport = reader.prepare('SELECT report, accepted_at FROM accepted WHERE tenant = ? AND source = ? AND period = ?');
    const readStreamObject = fixtureReader(h);
    return {
        close: () => { reader.close(); writer.close(); },
        seed() {
            writer.prepare('INSERT INTO connection VALUES (?, ?, 1, 0)').run(tenant, source);
            const insert = writer.prepare('INSERT INTO job VALUES (?, ?, ?, ?, 1, 0, NULL)');
            for (const id of ['first', 'correction', 'newer']) insert.run(tenant, id, source, '2026-01');
        },
        run: createCurRefreshJobRunner({
            now: () => new Date(Date.UTC(2026, 1, 2, h.hour)),
            loadJob({tenant, jobId}) {
                const job = getJob.get(tenant, jobId);
                return job && {...job, outcome: JSON.parse(job.outcome)};
            },
            openSource(binding) {
                const config = readSource.get(binding.tenant, binding.source);
                return config && {...config, manifestKey, readStreamObject};
            },
            commitJob({tenant, jobId, source, period, generation, mappingVersion, snapshot}) {
                writer.exec('BEGIN IMMEDIATE');
                let outcome;
                try {
                    const job = getJob.get(tenant, jobId);
                    assert.deepEqual([job.source, job.period, job.generation, job.mappingVersion],
                        [source, period, generation, mappingVersion]);
                    if (job.outcome) {
                        outcome = JSON.parse(job.outcome);
                    } else {
                        const {assemblyId, revision, report, acceptedAt} = snapshot;
                        const published = accept.run(period, assemblyId, revision, JSON.stringify(report), acceptedAt,
                            tenant, source, generation, mappingVersion).changes;
                        outcome = published ? {status: 'completed', publishedAt: acceptedAt} : {status: 'canceled'};
                        writer.prepare('UPDATE job SET outcome = ? WHERE tenant = ? AND id = ?')
                            .run(JSON.stringify(outcome), tenant, jobId);
                    }
                    writer.exec('COMMIT');
                } catch (error) {
                    writer.exec('ROLLBACK');
                    throw error;
                }
                if (h.loseAck) throw new Error('lost acknowledgement');
                return outcome;
            },
        }),
        read: createPrivateBillingRead({
            resolveSession(request) {
                const token = request.headers.get('Authorization');
                if (token === 'Bearer fixture-a') return {tenantId: tenant};
                if (token === 'Bearer fixture-b') return {tenantId: 'other'};
            },
            readJob({tenant, jobId}) {
                const job = getStatus.get(tenant, jobId);
                return job && (JSON.parse(job.outcome) ?? {status: 'queued'});
            },
            readReport({tenant, source, period}) {
                const row = getReport.get(tenant, source, period);
                return row && {report: JSON.parse(row.report), acceptedAt: row.accepted_at};
            },
        }),
    };
}

test('an on-disk restart preserves a committed correction and a private historical job', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'atlas-restart-'));
    const file = join(dir, 'billing.sqlite');
    const h = {hour: 1, cloudCalls: 0};
    let host;
    const restart = () => {
        if (host) host.close();
        host = null;
        host = openTestHost(file, h);
    };
    const target = {kind: 'report', source, period: '2026-01'};
    const job = {kind: 'job', jobId: 'correction'};
    const read = (target, token = 'fixture-a') =>
        host.read(new Request('https://atlas.invalid', {headers: {Authorization: 'Bearer ' + token}}), target);
    const expectReport = async (total, hour) => {
        const response = await read(target);
        assert.equal(response.status, 200);
        assert.equal(response.headers.get('Cache-Control'), 'private, no-store');
        const body = await response.json();
        assert.equal(body.report.TotalBilledCost, total);
        assert.equal(body.lastSuccessfulSync, new Date(Date.UTC(2026, 1, 2, hour)).toISOString());
        return body.lastSuccessfulSync;
    };
    try {
        restart();
        host.seed();
        await host.run({tenant, jobId: 'first'});
        await expectReport(100, 1);
        h.hour = 2;
        h.loseAck = true;
        await assert.rejects(host.run({tenant, jobId: job.jobId}), /lost acknowledgement/);

        restart();
        h.loseAck = false;
        const correctedAt = await expectReport(105, 2);
        assert.deepEqual(await (await read(job)).json(), {status: 'completed'});
        assert.equal((await read(target, 'fixture-b')).status, 404);
        assert.equal((await read(job, 'fixture-b')).status, 404);
        assert.equal((await read(target, 'invalid')).status, 401);
        const expected = {status: 'completed', publishedAt: correctedAt};
        let downloads = h.cloudCalls;
        assert.deepEqual(await host.run({tenant, jobId: job.jobId}), expected);
        assert.equal(h.cloudCalls, downloads);

        h.hour = 3;
        await host.run({tenant, jobId: 'newer'});
        assert.ok(h.cloudCalls > downloads);
        restart();
        downloads = h.cloudCalls;
        assert.deepEqual(await host.run({tenant, jobId: job.jobId}), expected);
        assert.equal(h.cloudCalls, downloads);
        await expectReport(110, 3);
    } finally {
        host?.close();
        rmSync(dir, {recursive: true, force: true});
    }
});
