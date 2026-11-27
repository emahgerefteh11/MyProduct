'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {createPrivateBillingWrite} = require('../server/private_billing_write.cjs');

const origin = 'https://atlas.invalid';
const refresh = Object.freeze({kind: 'refresh', source: 'aws', period: '2026-01'});
const disconnect = Object.freeze({kind: 'disconnect', source: 'aws', generation: 1});

function request({method = 'POST', site = origin, auth = 'operator-a', csrf = 'fixture-a', requestId = 'once'} = {}) {
    const headers = {Authorization: auth, 'X-CSRF-Token': csrf};
    if (site != null) headers.Origin = site;
    if (requestId != null) headers['Idempotency-Key'] = requestId;
    return new Request(origin + '/billing?tenant=tenant-b', {method, headers});
}

function fixture() {
    const h = {calls: [], jobs: new Map(), connections: new Map([['tenant-a:aws', {generation: 1, active: true}]])};
    h.handle = createPrivateBillingWrite({
        origin,
        resolveSession(request) {
            switch (request.headers.get('Authorization')) {
                case 'operator-a': return {tenantId: 'tenant-a', canManageBilling: true, nonce: 'fixture-a'};
                case 'operator-b': return {tenantId: 'tenant-b', canManageBilling: true, nonce: 'fixture-b'};
                case 'viewer': return {tenantId: 'tenant-a', canManageBilling: false, nonce: 'fixture-a'};
                default: return null;
            }
        },
        verifyIntent: (request, session) => request.headers.get('X-CSRF-Token') === session.nonce,
        enqueueRefresh({tenant, source, period, requestId}) {
            h.calls.push('queue');
            const connection = h.connections.get(tenant + ':' + source);
            if (!connection?.active) return null;
            const token = tenant + ':' + requestId;
            const previous = h.jobs.get(token);
            if (previous) {
                if (previous.source !== source || previous.period !== period || previous.generation !== connection.generation) {
                    return {status: 'conflict'};
                }
                return previous;
            }
            const job = {status: 'accepted', tenant, source, period, jobId: 'job-' + (h.jobs.size + 1),
                generation: connection.generation, privateField: 'never returned'};
            h.jobs.set(token, job);
            return job;
        },
        disconnectSource({tenant, source, generation}) {
            h.calls.push('disconnect');
            const binding = tenant + ':' + source;
            const connection = h.connections.get(binding);
            if (!connection) return null;
            if (!connection.active || connection.generation !== generation) return {status: 'conflict'};
            h.connections.set(binding, {generation: generation + 1, active: false});
            return {status: 'disconnected', tenant, source};
        },
    });
    return h;
}

test('changes require POST, the configured origin, and current session-bound intent and permission', async () => {
    const h = fixture();
    let response = await h.handle(request({method: 'GET'}), refresh);
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('Allow'), 'POST');
    for (const site of ['https://other.invalid', null]) {
        assert.equal((await h.handle(request({site}), refresh)).status, 403);
    }
    assert.equal((await h.handle(request({auth: 'revoked'}), refresh)).status, 401);
    assert.equal((await h.handle(request({auth: 'viewer'}), refresh)).status, 403);
    assert.equal((await h.handle(request({csrf: 'fixture-b'}), disconnect)).status, 403);
    assert.equal((await h.handle(request({requestId: null}), refresh)).status, 400);
    assert.equal((await h.handle(request(), {...disconnect, generation: '1'})).status, 404);
    assert.equal((await h.handle(request(), {...refresh, period: '9999-01'})).status, 404);
    assert.deepEqual(h.calls, []);
});

test('a refresh is scoped to the session, returns only a safe job reference, and deduplicates a retry', async () => {
    const h = fixture();
    const response = await h.handle(request(), {...refresh, tenant: 'tenant-b', bucket: 'untrusted'});
    assert.equal(response.status, 202);
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store');
    const first = await response.json();
    assert.deepEqual(first, {status: 'accepted', jobId: 'job-1'});
    assert.equal(h.jobs.get('tenant-a:once').tenant, 'tenant-a');
    assert.deepEqual(await (await h.handle(request(), refresh)).json(), first);
    assert.equal((await h.handle(request(), {...refresh, period: '2026-02'})).status, 409);
    assert.equal((await h.handle(request({auth: 'operator-b', csrf: 'fixture-b'}), refresh)).status, 404);
    assert.equal(h.jobs.size, 1);
});

test('a foreign or stale disconnect cannot disable the selected tenant connection', async () => {
    const h = fixture();
    assert.equal((await h.handle(request({auth: 'operator-b', csrf: 'fixture-b'}), disconnect)).status, 404);
    assert.deepEqual(h.connections.get('tenant-a:aws'), {generation: 1, active: true});
    const stopped = await h.handle(request(), disconnect);
    assert.equal(stopped.status, 200);
    assert.deepEqual(await stopped.json(), {status: 'disconnected'});
    assert.deepEqual(h.connections.get('tenant-a:aws'), {generation: 2, active: false});
    assert.equal((await h.handle(request(), refresh)).status, 404);
    h.connections.set('tenant-a:aws', {generation: 3, active: true});
    assert.equal((await h.handle(request(), disconnect)).status, 409);
    assert.deepEqual(h.connections.get('tenant-a:aws'), {generation: 3, active: true});
});

test('the storage boundary can deny an action after the session was resolved', async () => {
    const session = {tenantId: 'tenant-a', canManageBilling: true};
    const refuse = (scope, verifiedSession) => {
        assert.equal(scope.tenant, session.tenantId);
        assert.equal(verifiedSession, session);
        return {status: 'denied'};
    };
    const handler = createPrivateBillingWrite({
        origin, resolveSession: () => session, verifyIntent: () => true,
        enqueueRefresh: refuse, disconnectSource: refuse,
    });
    assert.equal((await handler(request(), refresh)).status, 403);
    assert.equal((await handler(request(), disconnect)).status, 403);
});

test('write dependency failures and a mismatched job do not expose internal details', async () => {
    const handler = createPrivateBillingWrite({
        origin, resolveSession: () => ({tenantId: 'tenant-a', canManageBilling: true}), verifyIntent: () => true,
        enqueueRefresh: () => ({status: 'accepted', tenant: 'tenant-b', source: 'aws', period: '2026-01', jobId: 'job-x'}),
        disconnectSource() { throw new Error('internal database failure'); },
    });
    for (const target of [refresh, disconnect]) {
        const response = await handler(request(), target);
        assert.equal(response.status, 503);
        assert.deepEqual(await response.json(), {error: 'Private billing is unavailable. Try again.'});
    }
});
