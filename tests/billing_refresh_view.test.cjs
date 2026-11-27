'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {createBillingRefreshView, decodeBillingActionResponse} = require('../billing_refresh_view.js');

const manager = {canManageBilling: true};
const jan = {source: 'aws', period: '2026-01', generation: 1};
const feb = {...jan, period: '2026-02'};
const first = {report: {sampleData: false, TotalBilledCost: 100}, lastSuccessfulSync: '2026-02-02T01:00:00.000Z'};
const corrected = {report: {sampleData: false, TotalBilledCost: 105}, lastSuccessfulSync: '2026-02-02T02:00:00.000Z'};
const freshScope = {report: {sampleData: false, TotalBilledCost: 40}, lastSuccessfulSync: '2026-03-02T01:00:00.000Z'};
const pending = () => {
    let resolve;
    const promise = new Promise(done => {resolve = done;});
    return {promise, resolve};
};

function setup(deps = {}) {
    const h = {states: [], submits: [], polls: [], reports: []};
    let issued = 0;
    h.view = createBillingRefreshView({
        submit: async (scope, options) => {h.submits.push([scope, options]); return {status: 'accepted', jobId: 'job-1'};},
        readJob: async (...request) => {h.polls.push(request); return {status: h.jobStatus};},
        readReport: async (...request) => {h.reports.push(request); return corrected;},
        disconnect: async () => ({status: 'disconnected'}),
        render: state => h.states.push(state),
        newRequestId: () => 'refresh-' + (++issued),
        ...deps,
    });
    h.last = () => h.states.at(-1);
    return h;
}

test('accepted report and timestamp only change after a completed job and fresh report read', async () => {
    const h = setup();
    h.view.select({...jan, tenant: 'never-sent'}, first, manager);
    await h.view.refresh();
    assert.deepEqual(h.submits[0][0], jan);
    assert.deepEqual(h.last(), {status: 'queued', accepted: first});
    h.jobStatus = 'running';
    await h.view.check();
    assert.deepEqual(h.last(), {status: 'running', accepted: first});
    assert.equal(h.reports.length, 0);
    h.jobStatus = 'completed';
    await h.view.check();
    assert.deepEqual(h.states.at(-2), {status: 'reading-report', accepted: first});
    assert.deepEqual(h.last(), {status: 'completed', accepted: corrected});
    assert.equal(h.reports.length, 1);
    await h.view.refresh();
    assert.notEqual(h.submits[0][1].requestId, h.submits[1][1].requestId);
});

test('a lost response retries the same action and a failed report fetch does not resubmit', async () => {
    const requestIds = [];
    const h = setup({
        submit: async (_scope, {requestId}) => {
            requestIds.push(requestId);
            if (requestIds.length === 1) throw new Error('lost response');
            return {status: 'accepted', jobId: 'one-durable-job'};
        },
        readReport: async () => {
            if (h.reloadBlocked) throw new Error('read unavailable');
            return corrected;
        },
    });
    h.view.select(jan, first, manager);
    await h.view.refresh();
    assert.deepEqual(h.last(), {status: 'retry-submit', accepted: first});
    await h.view.refresh();
    assert.deepEqual(requestIds, ['refresh-1', 'refresh-1']);
    h.jobStatus = 'completed';
    h.reloadBlocked = true;
    await h.view.check();
    assert.deepEqual(h.last(), {status: 'retry-status', accepted: first});
    h.reloadBlocked = false;
    await h.view.check();
    assert.deepEqual(h.last(), {status: 'completed', accepted: corrected});
    assert.equal(h.polls.length, 1);
    assert.equal(requestIds.length, 2);
});

test('a stale poll and a stale follow-up report cannot overwrite a new selection', async () => {
    const poll = pending();
    const report = pending();
    let pollSignal, reportSignal;
    const h = setup({
        readJob: (_id, {signal}) => {
            pollSignal = signal;
            return h.suspendPoll ? poll.promise : {status: 'completed'};
        },
        readReport: (_scope, {signal}) => {reportSignal = signal; return report.promise;},
    });
    h.view.select(jan, first, manager);
    await h.view.refresh();
    h.suspendPoll = true;
    const oldPoll = h.view.check();
    h.view.select(feb, freshScope, manager);
    assert.equal(pollSignal.aborted, true);
    poll.resolve({status: 'completed'});
    await oldPoll;
    assert.equal(reportSignal, undefined);
    assert.deepEqual(h.last(), {status: 'idle', accepted: freshScope});

    h.suspendPoll = false;
    h.view.select(jan, first, manager);
    await h.view.refresh();
    const oldReport = h.view.check();
    await Promise.resolve();
    assert.ok(reportSignal);
    h.view.select(feb, freshScope, manager);
    assert.equal(reportSignal.aborted, true);
    report.resolve(corrected);
    await oldReport;
    assert.deepEqual(h.last(), {status: 'idle', accepted: freshScope});
});

test('disconnect fences a late completion without removing an authorized accepted report', async () => {
    const report = pending();
    let reportSignal;
    const h = setup({
        readJob: async () => ({status: 'completed'}),
        readReport: (_scope, {signal}) => {reportSignal = signal; return report.promise;},
    });
    h.view.select(jan, first, manager);
    await h.view.refresh();
    const checking = h.view.check();
    await Promise.resolve();
    assert.ok(reportSignal);
    await h.view.stop();
    assert.equal(reportSignal.aborted, true);
    report.resolve(corrected);
    await checking;
    assert.deepEqual(h.last(), {status: 'disconnected', accepted: first});
    await h.view.refresh();
    assert.equal(h.submits.length, 1);
});

test('canceled jobs keep the last amount; authorization loss clears it', async () => {
    const h = setup();
    h.view.select(jan, first, manager);
    await h.view.refresh();
    h.jobStatus = 'canceled';
    await h.view.check();
    assert.deepEqual(h.last(), {status: 'canceled', accepted: first});
    await h.view.refresh();
    h.jobStatus = 'denied';
    await h.view.check();
    assert.deepEqual(h.last(), {status: 'hidden', accepted: null});
});

test('an unconfirmed disconnect keeps the accepted report but blocks new actions until reload', async () => {
    const h = setup({disconnect: async () => {throw new Error('disconnected or response lost');}});
    h.view.select(jan, first, manager);
    await h.view.stop();
    assert.deepEqual(h.last(), {status: 'disconnect-unconfirmed', accepted: first});
    await h.view.refresh();
    assert.equal(h.submits.length, 0);
    h.view.select(jan, first, manager);
    await h.view.refresh();
    assert.equal(h.submits.length, 1);
});

const refusal = status => new Response('untrusted provider details', {status});

test('a write 403 requires an authorized fresh read before showing a report to a viewer', async () => {
    const read = pending();
    let reads = 0, submits = 0;
    const h = setup({
        submit: async () => {submits++; return decodeBillingActionResponse(refusal(403));},
        readSelection: async () => {reads++; return read.promise;},
    });
    h.view.select(jan, first, manager);
    const write = h.view.refresh();
    await new Promise(setImmediate);
    assert.equal(reads, 1);
    assert.deepEqual(h.last(), {status: 'verifying-access', accepted: null});
    read.resolve({scope: jan, accepted: corrected, canManageBilling: false});
    await write;
    assert.deepEqual(h.last(), {status: 'read-only', accepted: corrected});
    await h.view.refresh();
    await h.view.stop();
    assert.equal(submits, 1);
    h.view.select(jan, corrected);
    assert.deepEqual(h.last(), {status: 'read-only', accepted: corrected});
});

test('a revoked session hides private data, whether detected at write or on revalidation', async () => {
    let code = 403, reads = 0;
    const h = setup({
        submit: () => decodeBillingActionResponse(refusal(code)),
        readSelection: async () => {reads++; return {status: 'denied'};},
    });
    h.view.select(jan, first, manager);
    await h.view.refresh();
    assert.deepEqual(h.last(), {status: 'hidden', accepted: null});
    assert.equal(reads, 1);
    code = 401;
    h.view.select(jan, corrected, manager);
    await h.view.refresh();
    assert.deepEqual(h.last(), {status: 'hidden', accepted: null});
    assert.equal(reads, 1);
});

test('a stale disconnect reloads the generation without retrying the disconnect', async () => {
    const disconnects = [];
    const newer = {...jan, generation: 2};
    const h = setup({
        async disconnect(scope) {disconnects.push(scope); return decodeBillingActionResponse(refusal(409));},
        readSelection: async () => ({scope: newer, accepted: corrected, canManageBilling: true}),
    });
    h.view.select(jan, first, manager);
    await h.view.stop();
    assert.deepEqual(disconnects, [jan]);
    assert.deepEqual(h.last(), {status: 'selection-reloaded', accepted: corrected});
    await h.view.refresh();
    assert.deepEqual(h.submits[0][0], newer);
});

test('an old scope cannot restore a report from a delayed permission recheck', async () => {
    const read = pending();
    let signal;
    const h = setup({
        submit: () => decodeBillingActionResponse(refusal(403)),
        readSelection: (_scope, options) => {signal = options.signal; return read.promise;},
    });
    h.view.select(jan, first, manager);
    const old = h.view.refresh();
    await new Promise(setImmediate);
    assert.ok(signal);
    h.view.select(feb, freshScope, manager);
    assert.equal(signal.aborted, true);
    read.resolve({scope: jan, accepted: corrected, canManageBilling: false});
    await old;
    assert.deepEqual(h.last(), {status: 'idle', accepted: freshScope});
});
