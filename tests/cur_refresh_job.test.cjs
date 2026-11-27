'use strict';

const assert = require('node:assert/strict');
const {mkdtempSync, rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const test = require('node:test');
const {createCurRefreshJobRunner} = require('../server/cur_refresh_job.cjs');

const {tenant, source, origin, manifestFor, manifestKey, fixtureReader, openTestHost} = require('./fixtures/cur_test_host.cjs');

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

test('a live SQLite backup restores the report and its job outcome at the same checkpoint', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'atlas-backup-'));
    const file = join(dir, 'active.sqlite');
    const snapshot = join(dir, 'restored.sqlite');
    const h = {hour: 1, cloudCalls: 0};
    const report = {kind: 'report', source, period: '2026-01'};
    const run = (host, jobId) => host.run({tenant, jobId});
    const read = (host, target, token = 'fixture-a') =>
        host.read(new Request('https://atlas.invalid', {headers: {Authorization: 'Bearer ' + token}}), target);
    let active, restored;
    try {
        active = openTestHost(file, h);
        active.seed();
        await run(active, 'first');
        h.hour = 2;
        const correction = await run(active, 'correction');
        assert.ok((await active.backupTo(snapshot)) > 0);

        h.hour = 3;
        const newer = await run(active, 'newer');
        assert.equal((await (await read(active, report)).json()).report.TotalBilledCost, 110);

        // Never restore over the active storage. Post-snapshot $110 must not appear
        // in the .sqlite backup or be implied by the restored job table.
        restored = openTestHost(snapshot, h);
        const result = await read(restored, report);
        assert.equal(result.status, 200);
        assert.equal(result.headers.get('Cache-Control'), 'private, no-store');
        const body = await result.json();
        assert.equal(body.report.TotalBilledCost, 105);
        assert.equal(body.lastSuccessfulSync, correction.publishedAt);
        assert.notEqual(body.lastSuccessfulSync, newer.publishedAt);
        const restoredJob = {kind: 'job', jobId: 'correction'};
        assert.deepEqual(await (await read(restored, restoredJob)).json(), {status: 'completed'});
        assert.deepEqual(await (await read(restored, {kind: 'job', jobId: 'newer'})).json(), {status: 'queued'});

        assert.equal((await read(restored, report, 'fixture-b')).status, 404);
        assert.equal((await read(restored, restoredJob, 'fixture-b')).status, 404);
        assert.equal((await read(restored, report, 'invalid')).status, 401);
        const calls = h.cloudCalls;
        assert.deepEqual(await run(restored, 'correction'), correction);
        assert.equal(h.cloudCalls, calls);
        assert.equal((await (await read(active, report)).json()).report.TotalBilledCost, 110);
    } finally {
        restored?.close();
        active?.close();
        rmSync(dir, {recursive: true, force: true});
    }
});

test('authorized actions queue one durable job and disconnect fences an in-flight revision', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'atlas-actions-'));
    const file = join(dir, 'billing.sqlite');
    const h = {hour: 2, cloudCalls: 0};
    let host;
    const refresh = {kind: 'refresh', source, period: '2026-01'};
    const report = {kind: 'report', source, period: refresh.period};
    const disconnect = {kind: 'disconnect', source, generation: 1};
    const read = (target, actor = 'fixture-a') => host.read(new Request(origin, {
        headers: {Authorization: 'Bearer ' + actor},
    }), target);
    const write = (target, id, actor = 'fixture-a') => host.write(new Request(origin, {
        method: 'POST', headers: {Origin: origin, Authorization: 'Bearer ' + actor,
            'X-CSRF-Token': actor, 'Idempotency-Key': id},
    }), target);
    try {
        host = openTestHost(file, h);
        host.seed({jobs: false});
        assert.equal((await write(refresh, 'foreign', 'fixture-b')).status, 404);
        assert.equal((await write(refresh, 'revoked', 'revoked')).status, 401);
        h.beforeWrite = () => host.setManage(false);
        assert.equal((await write(refresh, 'correction')).status, 403);
        host.setManage(true);
        h.beforeWrite = null;

        const queued = await write(refresh, 'correction');
        assert.equal(queued.status, 202);
        const job = await queued.json();
        const readJob = id => ({kind: 'job', jobId: id});
        assert.deepEqual(await (await read(readJob(job.jobId))).json(), {status: 'queued'});
        assert.equal((await read(readJob(job.jobId), 'fixture-b')).status, 404);
        assert.equal(h.cloudCalls, 0, 'queueing must not read S3');
        host.close();
        host = null;
        host = openTestHost(file, h);
        assert.deepEqual(await (await write(refresh, 'correction')).json(), job);
        assert.equal(h.cloudCalls, 0);
        const completed = await host.run({tenant, jobId: job.jobId});
        assert.equal(completed.status, 'completed');
        const reportBefore = await (await read(report)).json();
        assert.equal(reportBefore.report.TotalBilledCost, 105);
        assert.equal(reportBefore.lastSuccessfulSync, completed.publishedAt);

        const next = await write(refresh, 'newer');
        assert.equal(next.status, 202);
        const {jobId} = await next.json();
        h.hour = 3;
        h.hook = async () => {
            assert.equal((await write(disconnect, 'foreign-stop', 'fixture-b')).status, 404);
            assert.equal((await write(disconnect, 'stop')).status, 200);
        };
        assert.deepEqual(await host.run({tenant, jobId}), {status: 'canceled'});
        assert.deepEqual(await (await read(readJob(jobId))).json(), {status: 'canceled'});
        assert.deepEqual(await (await read(report)).json(), reportBefore);
        assert.equal((await write(refresh, 'closed')).status, 404);
        const fetches = h.cloudCalls;
        assert.deepEqual(await host.run({tenant, jobId: job.jobId}), completed);
        assert.equal(h.cloudCalls, fetches);
    } finally {
        host?.close();
        rmSync(dir, {recursive: true, force: true});
    }
});

test('accepted overview and detail stay on the same scoped revision even at the same clock time', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'atlas-revision-detail-'));
    const h = {hour: 2, cloudCalls: 0, acceptanceHour: 2};
    let host;
    try {
        host = openTestHost(join(dir, 'reports.sqlite'), h, {acceptedTokenKey: Buffer.alloc(32, 18)});
        host.seed();
        assert.equal((await host.run({tenant, jobId: 'correction'})).status, 'completed');
        const read = async (target, actor = 'fixture-a') => {
            const request = new Request(origin + '/private', {headers: {Authorization: 'Bearer ' + actor}});
            const result = await host.read(request, {source, period: '2026-01', ...target});
            assert.equal(result.headers.get('Cache-Control'), 'private, no-store');
            return {status: result.status, body: await result.json()};
        };
        const current = {kind: 'overview'};
        const first = await read(current);
        assert.equal(first.status, 200);
        assert.equal(first.body.overview.totals.amount, 105);
        assert.doesNotMatch(JSON.stringify(first.body), /20260202T020000Z|fixture\/export/);
        const requestDetail = acceptedRevision => ({kind: 'detail', dimension: 'products', offset: 0, acceptedRevision});
        const olderDetail = requestDetail(first.body.acceptedRevision);
        const old = await read(olderDetail);
        assert.equal(old.status, 200);
        assert.ok(old.body.detail.entries.some(row => row.amount === 35));
        h.hour = 3;
        assert.equal((await host.run({tenant, jobId: 'newer'})).status, 'completed');
        const stale = await read(olderDetail);
        assert.deepEqual(stale, {status: 409, body: {error: 'Billing report changed. Reload the overview.'}});
        const fresh = await read(current);
        assert.equal(fresh.body.overview.totals.amount, 110);
        assert.equal(fresh.body.overview.lastSuccessfulSync, first.body.overview.lastSuccessfulSync);
        assert.notEqual(fresh.body.acceptedRevision, first.body.acceptedRevision);
        const freshDetail = requestDetail(fresh.body.acceptedRevision);
        assert.ok((await read(freshDetail)).body.detail.entries.some(row => row.amount === 40));
        assert.equal((await read({...freshDetail, offset: -1})).status, 400);
        assert.equal((await read({...freshDetail, dimension: 'constructor'})).status, 400);
        assert.equal((await read({...freshDetail, offset: 100_000})).status, 404);
        for (const query of [olderDetail, freshDetail]) {
            assert.equal((await read(query, 'fixture-b')).status, 404);
            assert.equal((await read(query, 'revoked')).status, 401);
        }
        host.revokeRead();
        assert.equal((await read(current)).status, 404);
        assert.equal((await read(olderDetail)).status, 404);
        assert.equal((await read(freshDetail)).status, 404);
    } finally {
        host?.close();
        rmSync(dir, {recursive: true, force: true});
    }
});

test('a stored billing overview commits and rolls back together with its report and job result', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'atlas-materialized-overview-'));
    const file = join(dir, 'report.sqlite');
    const h = {hour: 1, cloudCalls: 0};
    let host;
    try {
        host = openTestHost(file, h);
        host.seed();
        const target = {source, period: '2026-01'};
        const session = actor => new Request(origin + '/private', {headers: {Authorization: 'Bearer ' + actor}});
        const current = () => host.storedOverview(session('fixture-a'), target);
        const request = session('fixture-a');
        const job = id => host.read(request, {kind: 'job', jobId: id}).then(response => response.json());
        assert.equal(current(), null);
        assert.equal((await host.run({tenant, jobId: 'first'})).status, 'completed');
        const first = current();
        assert.equal(first.overview.totals.amount, 100);
        h.hour = 2;
        h.failBeforeJobResult = true;
        await assert.rejects(host.run({tenant, jobId: 'correction'}), /synthetic outcome failure/);
        assert.deepEqual(current(), first);
        assert.equal((await job('correction')).status, 'queued');
        h.failBeforeJobResult = false;
        assert.equal((await host.run({tenant, jobId: 'correction'})).status, 'completed');
        const revised = current();
        assert.equal(revised.overview.totals.amount, 105);
        assert.equal(revised.overview.allocation.unallocatedProductCost, 35);
        assert.notEqual(revised.revision, first.revision);
        const full = await host.read(request, {kind: 'report', ...target}).then(response => response.json());
        assert.equal(revised.overview.totals.amount, full.report.TotalBilledCost);
        assert.equal(revised.overview.lastSuccessfulSync, full.lastSuccessfulSync);
        host.close();
        host = openTestHost(file, h);
        assert.deepEqual(current(), revised);
        assert.equal((await job('correction')).status, 'completed');
        const calls = h.cloudCalls;
        assert.equal((await host.run({tenant, jobId: 'correction'})).status, 'completed');
        assert.equal(h.cloudCalls, calls);
        assert.equal(host.storedOverview(session('fixture-b'), target), null);
        host.revokeRead();
        assert.equal(current(), null);
    } finally {
        host?.close();
        rmSync(dir, {recursive: true, force: true});
    }
});
