'use strict';

const assert = require('node:assert/strict');
const {randomUUID} = require('node:crypto');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {backup, DatabaseSync} = require('node:sqlite');
const {Readable} = require('node:stream');
const {gzipSync} = require('node:zlib');
const {bindKeptCurS3Reader} = require('../../server/scoped_cur_s3.cjs');
const {createCurRefreshJobRunner} = require('../../server/cur_refresh_job.cjs');
const {createPrivateBillingRead} = require('../../server/private_billing_read.cjs');
const {createPrivateBillingWrite} = require('../../server/private_billing_write.cjs');

const tenant = 'test-tenant', source = 'aws', origin = 'https://atlas.invalid';
const manifestKey = 'fixture/export/20260101-20260201/atlas-Manifest.json';
const csv = readFileSync(join(__dirname, 'cur', 'synthetic.csv'), 'utf8');
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

// A test-only host: no HTTP routes, real sessions, AWS credentials or implied
// production database choice. Independent connections model the reader/worker split.
function openTestHost(file, h, {hostOrigin = origin} = {}) {
    const writer = new DatabaseSync(file);
    writer.exec(`
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = FULL;
        CREATE TABLE IF NOT EXISTS job (
            tenant TEXT, id TEXT, source TEXT, period TEXT, generation INTEGER,
            mapping INTEGER, outcome TEXT, request_id TEXT,
            PRIMARY KEY (tenant, id), UNIQUE (tenant, request_id)
        );
        CREATE TABLE IF NOT EXISTS connection (
            tenant TEXT, source TEXT, generation INTEGER, mapping INTEGER, active INTEGER,
            PRIMARY KEY (tenant, source)
        );
        CREATE TABLE IF NOT EXISTS grant_state (
            tenant TEXT, actor TEXT, can_manage INTEGER, PRIMARY KEY (tenant, actor)
        );
        CREATE TABLE IF NOT EXISTS accepted (
            tenant TEXT, source TEXT, period TEXT, assembly TEXT, revision TEXT,
            mapping INTEGER, report TEXT, accepted_at TEXT, PRIMARY KEY (tenant, source, period)
        );
    `);
    assert.equal(writer.prepare('PRAGMA journal_mode').get().journal_mode, 'wal');
    const reader = new DatabaseSync(file, {readOnly: true});
    const findJob = db => db.prepare(`
        SELECT tenant, id AS jobId, source, period, generation, mapping AS mappingVersion, outcome
        FROM job WHERE tenant = ? AND id = ?
    `);
    const getJob = findJob(writer);
    const getStatus = findJob(reader);
    const readSource = writer.prepare('SELECT generation, mapping AS mappingVersion FROM connection WHERE tenant = ? AND source = ? AND active = 1');
    const accept = writer.prepare(`
        INSERT INTO accepted
        SELECT tenant, source, ?, ?, ?, mapping, ?, ? FROM connection
        WHERE tenant = ? AND source = ? AND generation = ? AND mapping = ? AND active = 1
        ON CONFLICT (tenant, source, period) DO UPDATE SET assembly = excluded.assembly,
            revision = excluded.revision, mapping = excluded.mapping, report = excluded.report,
            accepted_at = excluded.accepted_at
        WHERE excluded.assembly > accepted.assembly
           OR (excluded.assembly = accepted.assembly AND excluded.revision = accepted.revision
               AND excluded.mapping > accepted.mapping)
    `);
    const getReport = reader.prepare('SELECT report, accepted_at FROM accepted WHERE tenant = ? AND source = ? AND period = ?');
    const readStreamObject = fixtureReader(h);
    const canManage = writer.prepare('SELECT can_manage FROM grant_state WHERE tenant = ? AND actor = ?');
    const resolveSession = request => {
        const token = request.headers.get('Authorization');
        const actor = token === 'Bearer fixture-a' ? 'fixture-a' : token === 'Bearer fixture-b' ? 'fixture-b' : null;
        if (!actor) return null;
        const scope = actor === 'fixture-a' ? tenant : 'other';
        return {actor, tenantId: scope, canManageBilling: canManage.get(scope, actor)?.can_manage === 1};
    };
    function transact(operation) {
        writer.exec('BEGIN IMMEDIATE');
        try {
            const result = operation();
            writer.exec('COMMIT');
            return result;
        } catch (error) {
            writer.exec('ROLLBACK');
            throw error;
        }
    }
    return {
        resolveSession,
        connectionState: (tenant, source) => writer.prepare(
            'SELECT generation, active FROM connection WHERE tenant = ? AND source = ?'
        ).get(tenant, source),
        advanceGeneration: () => writer.prepare(
            'UPDATE connection SET generation = generation + 1 WHERE tenant = ? AND source = ?'
        ).run(tenant, source),
        close: () => { reader.close(); writer.close(); },
        backupTo: destination => backup(reader, destination),
        seed({jobs = true} = {}) {
            writer.prepare('INSERT INTO connection VALUES (?, ?, 1, 0, 1)').run(tenant, source);
            const grant = writer.prepare('INSERT INTO grant_state VALUES (?, ?, 1)');
            grant.run(tenant, 'fixture-a');
            grant.run('other', 'fixture-b');
            if (!jobs) return;
            const insert = writer.prepare('INSERT INTO job VALUES (?, ?, ?, ?, 1, 0, NULL, NULL)');
            for (const id of ['first', 'correction', 'newer']) insert.run(tenant, id, source, '2026-01');
        },
        write: createPrivateBillingWrite({
            origin: hostOrigin, resolveSession,
            async verifyIntent(request, session) {
                const matches = request.headers.get('X-CSRF-Token') === session.actor;
                if (h.beforeWrite) await h.beforeWrite();
                return matches;
            },
            enqueueRefresh({tenant, source, period, requestId}, session) {
                return transact(() => {
                    if (canManage.get(tenant, session.actor)?.can_manage !== 1) return {status: 'denied'};
                    const active = readSource.get(tenant, source);
                    if (!active || period !== '2026-01') return null;
                    const previous = writer.prepare(`
                        SELECT id AS jobId, source, period, generation, mapping
                        FROM job WHERE tenant = ? AND request_id = ?
                    `).get(tenant, requestId);
                    if (previous && (previous.source !== source || previous.period !== period ||
                        previous.generation !== active.generation || previous.mapping !== active.mappingVersion)) {
                        return {status: 'conflict'};
                    }
                    const jobId = previous?.jobId ?? randomUUID();
                    if (!previous) {
                        writer.prepare('INSERT INTO job VALUES (?, ?, ?, ?, ?, ?, NULL, ?)')
                            .run(tenant, jobId, source, period, active.generation, active.mappingVersion, requestId);
                    }
                    return {status: 'accepted', tenant, source, period, jobId};
                });
            },
            disconnectSource({tenant, source, generation}, session) {
                return transact(() => {
                    if (canManage.get(tenant, session.actor)?.can_manage !== 1) return {status: 'denied'};
                    const active = readSource.get(tenant, source);
                    if (!active) return null;
                    if (active.generation !== generation) return {status: 'conflict'};
                    writer.prepare(`
                        UPDATE connection SET generation = generation + 1, active = 0
                        WHERE tenant = ? AND source = ? AND generation = ?
                    `).run(tenant, source, generation);
                    return {status: 'disconnected', tenant, source};
                });
            },
        }),
        setManage: enabled => writer.prepare('UPDATE grant_state SET can_manage = ? WHERE tenant = ? AND actor = ?')
            .run(Number(enabled), tenant, 'fixture-a'),
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
                const outcome = transact(() => {
                    const job = getJob.get(tenant, jobId);
                    assert.deepEqual([job.source, job.period, job.generation, job.mappingVersion],
                        [source, period, generation, mappingVersion]);
                    if (job.outcome) return JSON.parse(job.outcome);
                    const {assemblyId, revision, report, acceptedAt} = snapshot;
                    const published = accept.run(period, assemblyId, revision, JSON.stringify(report), acceptedAt,
                        tenant, source, generation, mappingVersion).changes;
                    const result = published ? {status: 'completed', publishedAt: acceptedAt} : {status: 'canceled'};
                    writer.prepare('UPDATE job SET outcome = ? WHERE tenant = ? AND id = ?')
                        .run(JSON.stringify(result), tenant, jobId);
                    return result;
                });
                if (h.loseAck) throw new Error('lost acknowledgement');
                return outcome;
            },
        }),
        read: createPrivateBillingRead({
            resolveSession,
            readJob({tenant, jobId}) {
                const job = getStatus.get(tenant, jobId);
                return job && (JSON.parse(job.outcome) ?? {status: h.running === jobId ? 'running' : 'queued'});
            },
            readReport({tenant, source, period}) {
                const row = getReport.get(tenant, source, period);
                return row && {report: JSON.parse(row.report), acceptedAt: row.accepted_at};
            },
        }),
    };
}

module.exports = {tenant, source, origin, manifestFor, manifestKey, fixtureReader, openTestHost};
