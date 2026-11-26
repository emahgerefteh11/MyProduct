'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const {createReportViewSession} = require('../report_view_session.js');

function setup() {
    const stored = {total: 100};
    const pending = [];
    const states = [];
    let uploadReads = 0;
    const view = createReportViewSession({
        readConnected: (selection, signal) => new Promise((resolve, reject) => pending.push({selection, signal, resolve, reject})),
        readUpload: async () => { uploadReads++; return stored; },
        render: state => states.push(state),
    });
    return {view, pending, states, stored, current: () => states.at(-1), reads: () => uploadReads};
}

test('connected view does not read or overwrite uploads; failed read has no local fallback', async () => {
    const h = setup();
    let opened = h.view.openConnected('fixture-a');
    assert.equal(h.reads(), 0);
    h.pending[0].resolve({total: 105});
    await opened;
    assert.deepEqual(h.current(), {source: 'connected', status: 'ready', report: {total: 105}});
    opened = h.view.openConnected('fixture-a');
    h.pending[1].reject(new Error('private detail'));
    await opened;
    assert.deepEqual(h.current(), {source: 'connected', status: 'unavailable'});
    assert.equal(h.reads(), 0);
    await h.view.openUpload();
    assert.equal(h.reads(), 1);
    assert.deepEqual(h.current(), {source: 'upload', status: 'ready', report: {total: 100}});
});

test('switch and sign-out clear displayed data and reject late completion even if abort is ignored', async () => {
    const h = setup();
    const old = h.view.openConnected('fixture-a');
    const next = h.view.openConnected('fixture-b');
    assert.equal(h.pending[0].signal.aborted, true);
    assert.deepEqual(h.current(), {source: 'connected', status: 'loading'});
    h.pending[1].resolve({total: 50});
    await next;
    h.pending[0].resolve({total: 105});
    await old;
    assert.deepEqual(h.current().report, {total: 50});
    const late = h.view.openConnected('fixture-b');
    h.view.clear();
    assert.equal(h.pending[2].signal.aborted, true);
    h.pending[2].resolve({total: 50});
    await late;
    assert.deepEqual(h.current(), {status: 'hidden'});
});
