'use strict';

const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const test = require('node:test');
const {gzipSync} = require('node:zlib');
const {createLocalCurConnector} = require('../local_cur_connector.cjs');
const {createPrivateBillingRead} = require('../server/private_billing_read.cjs');
const {createReportViewSession} = require('../report_view_session.js');

const fixture = name => readFileSync(join(__dirname, 'fixtures', 'cur', name));
const manifest = JSON.parse(fixture('manifest.json'));
const source = 'fixture-aws';
const currentPeriod = '2026-01';
const scope = {source, start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z'};
const target = {kind: 'report', source, period: currentPeriod};
const request = (session, extra = {}) => new Request('https://atlas.invalid/private?tenant=tenant-a', {
    headers: {Authorization: session || '', 'X-Tenant': 'tenant-a'}, ...extra,
});

function setup() {
    const raw = fixture('synthetic.csv').toString();
    const objects = {
        'tenant-a': gzipSync(raw.replace(',30,USD,AmazonEC2,', ',35,USD,AmazonEC2,')),
        'tenant-b': gzipSync(raw),
    };
    const connector = createLocalCurConnector({
        readObject: ({tenant}) => objects[tenant],
        now: () => new Date('2026-02-02T02:05:00Z'),
    });
    for (const tenant of Object.keys(objects)) connector.refresh({tenant, source, manifest});
    const sessions = new Map([['test-a', 'tenant-a'], ['test-b', 'tenant-b']]);
    const lookups = [];
    const handle = createPrivateBillingRead({
        resolveSession: async req => {
            const actor = req.headers.get('Authorization');
            return {tenantId: sessions.get(actor), actor};
        },
        readReport: async query => {
            lookups.push(query);
            if (query.source === 'storage-error') throw new Error('provider-error-detail');
            return query.period === currentPeriod ? connector.get({...scope, source: query.source, tenant: query.tenant}) : null;
        },
        readJob: async query => {
            lookups.push(query);
            return query.tenant === 'tenant-a' && query.jobId === 'job-a'
                ? {status: 'completed', internal: 'provider-error-detail'} : null;
        },
    });
    return {handle, sessions, lookups, connector};
}

test('connector reports feed the view selected under the server session without touching the upload', async () => {
    const h = setup();
    let session = 'test-a', visible, localReads = 0;
    const view = createReportViewSession({
        readConnected: async selection => {
            const reply = await h.handle(request(session), selection);
            assert.equal(reply.headers.get('Cache-Control'), 'private, no-store');
            return reply.ok ? (await reply.json()).report : null;
        },
        readUpload: async () => { localReads++; return {TotalBilledCost: 100}; },
        render: state => {visible = state;},
    });
    await view.openConnected(target);
    assert.equal(visible.report.TotalBilledCost, 105);
    session = 'test-b';
    await view.openConnected({...target, tenantId: 'tenant-a'});
    assert.equal(visible.report.TotalBilledCost, 100);
    h.sessions.delete('test-b');
    await view.openConnected(target);
    assert.deepEqual(visible, {source: 'connected', status: 'unavailable'});
    assert.equal(localReads, 0);
    await view.openUpload();
    assert.equal(visible.report.TotalBilledCost, 100);
    assert.equal(localReads, 1);
    assert.deepEqual(h.lookups, [
        {tenant: 'tenant-a', actor: 'test-a', source, period: currentPeriod},
        {tenant: 'tenant-b', actor: 'test-b', source, period: currentPeriod},
    ]);
});

test('private read rejects unauthenticated requests, limits job data and hides storage errors', async () => {
    const {handle, lookups} = setup();
    const job = {kind: 'job', jobId: 'job-a'};
    const denied = await handle(request(null), target);
    assert.equal(denied.status, 401);
    assert.deepEqual(lookups, []);
    const own = await handle(request('test-a'), job);
    assert.equal(own.status, 200);
    assert.deepEqual(await own.json(), {status: 'completed'});
    const foreign = await handle(request('test-b'), job);
    const absent = await handle(request('test-b'), {...job, jobId: 'absent'});
    assert.deepEqual([foreign.status, await foreign.json()], [absent.status, await absent.json()]);
    const error = await handle(request('test-a'), {...target, source: 'storage-error'});
    assert.equal(error.status, 503);
    assert.doesNotMatch(await error.text(), /provider-error-detail/);
    const method = await handle(request('test-a', {method: 'POST'}), target);
    assert.equal(method.status, 405);
    assert.equal(method.headers.get('Allow'), 'GET');
    const invalid = await handle(request('test-a'), {...target, period: '2026-13'});
    assert.equal(invalid.status, 404);
    for (const reply of [denied, own, foreign, absent, error, method, invalid]) {
        assert.equal(reply.headers.get('Cache-Control'), 'private, no-store');
    }
});

test('private selection uses current scoped grant, connection generation, and accepted report', async () => {
    const scope = {tenant: 'tenant-a', source, period: currentPeriod};
    const target = {kind: 'selection', source, period: currentPeriod, tenant: 'untrusted'};
    const acceptedAt = '2026-02-02T02:05:00Z';
    const report = {sampleData: false, TotalBilledCost: 105};
    let selection = {...scope, active: true, canManageBilling: true, generation: 3,
        snapshot: {...scope, acceptedAt, report}, internal: 'never-return-this'};
    const seen = [];
    const handle = createPrivateBillingRead({
        resolveSession: req => req.headers.get('Authorization') === 'test-a'
            ? {tenantId: 'tenant-a', actor: 'alice', canManageBilling: true} : null,
        readSelection(query) {seen.push(query); return selection;},
    });
    const load = async (token = 'test-a', destination = target) => {
        const res = await handle(request(token), destination);
        assert.equal(res.headers.get('Cache-Control'), 'private, no-store');
        return [res.status, await res.json()];
    };
    assert.equal((await load('revoked'))[0], 401);
    assert.equal((await load('test-a', {...target, period: '2026-00'}))[0], 404);
    assert.deepEqual(seen, []);
    const scoped = {source, period: currentPeriod, generation: 3};
    const accepted = {report, lastSuccessfulSync: acceptedAt};
    assert.deepEqual(await load(), [200, {scope: scoped, connectionStatus: 'active', canManageBilling: true, accepted}]);
    assert.deepEqual(seen, [{...scope, actor: 'alice'}]);
    selection.canManageBilling = false;
    assert.deepEqual(await load(), [200, {scope: scoped, connectionStatus: 'active', canManageBilling: false, accepted}]);
    selection = {...selection, active: false, canManageBilling: true, generation: 4};
    assert.deepEqual(await load(), [200, {scope: {...scoped, generation: 4},
        connectionStatus: 'disconnected', canManageBilling: false, accepted}]);
    selection = {...selection, active: true, snapshot: null};
    assert.deepEqual(await load(), [200, {scope: {...scoped, generation: 4},
        connectionStatus: 'active', canManageBilling: true, accepted: null}]);
    for (const value of [
        {tenant: 'foreign'}, {source: 'wrong'}, {period: '2026-02'}, {generation: 0},
        {snapshot: {...scope, tenant: 'foreign', acceptedAt, report}},
        {snapshot: {...scope, source: 'wrong', acceptedAt, report}},
        {snapshot: {...scope, period: '2026-02', acceptedAt, report}},
        {snapshot: {...scope, acceptedAt: 'private-storage-detail', report}},
        {snapshot: {...scope, acceptedAt, report: {...report, sampleData: true}}},
    ]) {
        const previous = selection;
        selection = {...selection, ...value};
        assert.deepEqual(await load(), [503, {error: 'Private billing is unavailable. Try again.'}]);
        selection = previous;
    }
    selection = null;
    assert.deepEqual(await load(), [404, {error: 'Not found.'}]);
});
