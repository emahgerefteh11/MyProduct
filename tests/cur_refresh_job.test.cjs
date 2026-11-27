'use strict';

const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Readable} = require('node:stream');
const test = require('node:test');
const {gzipSync} = require('node:zlib');
const {bindKeptCurS3Reader} = require('../server/scoped_cur_s3.cjs');
const {createCurRefreshJobRunner} = require('../server/cur_refresh_job.cjs');

const tenant = 'test-tenant', source = 'aws';
const manifestKey = 'fixture/export/20260101-20260201/atlas-Manifest.json';
const csv = readFileSync(join(__dirname, 'fixtures', 'cur', 'synthetic.csv'), 'utf8');
const manifestFor = hour => {
    const assemblyId = '20260202T0' + hour + '0000Z';
    return {assemblyId, reportName: 'atlas',
        billingPeriod: {start: '20260101T000000.000Z', end: '20260201T000000.000Z'},
        reportKeys: ['fixture/export/20260101-20260201/' + assemblyId + '/atlas-00001.csv.gz']};
};

function setup() {
    const h = {hour: 1, opens: 0, committed: 0, cloudCalls: 0, loseAck: false, activeGeneration: 1,
        jobs: new Map(), latest: null, hook: null};
    h.seed = jobId => h.jobs.set(jobId, {tenant, jobId, source, period: '2026-01', generation: 1, mappingVersion: 0});
    const adapter = bindKeptCurS3Reader({
        tenant, source, manifestKey, bucket: 'fixture-cur', expectedBucketOwner: '111122223333',
        sendGetObject: async ({Key, IfMatch}) => {
            h.cloudCalls++;
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
    h.run = createCurRefreshJobRunner({
        now: () => new Date(Date.UTC(2026, 1, 2, h.hour)),
        loadJob: async ({tenant: scope, jobId}) => scope === tenant ? h.jobs.get(jobId) : null,
        openSource: async binding => {
            h.opens++;
            assert.equal(binding.tenant, tenant);
            assert.equal(binding.source, source);
            assert.equal(binding.period, '2026-01');
            return {generation: h.activeGeneration, mappingVersion: 0, manifestKey, readStreamObject: adapter};
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
