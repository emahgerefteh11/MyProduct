'use strict';

const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Readable} = require('node:stream');
const test = require('node:test');
const {gzipSync} = require('node:zlib');
const {LIMITS} = require('../billing_import.js');
const {createLocalCurConnector} = require('../local_cur_connector.cjs');
const {readBoundedS3Object} = require('../server/bounded_s3_read.cjs');

const fixture = name => readFileSync(join(__dirname, 'fixtures', 'cur', name));

test('counts actual bytes without ContentLength and closes streams that exceed the limit', async () => {
    const body = Readable.from([Buffer.from('ab'), Buffer.from('c')]);
    const accepted = await readBoundedS3Object({key: 'part', maxBytes: 3, getObject: async () => ({Body: body, ETag: '"example"'})});
    assert.deepEqual(accepted, {bytes: Buffer.from('abc'), etag: '"example"'});
    for (const ContentLength of [undefined, 100]) {
        const tooLarge = Readable.from([Buffer.from('abc'), Buffer.from('def')]);
        await assert.rejects(readBoundedS3Object({
            key: 'part', maxBytes: 4, getObject: async () => ({Body: tooLarge, ContentLength}),
        }), {code: 'OBJECT_TOO_LARGE'});
        assert.equal(tooLarge.destroyed, true);
    }
    await assert.rejects(readBoundedS3Object({
        key: 'part', maxBytes: 4,
        getObject: async () => ({Body: Readable.from([Buffer.from('a')]), ContentLength: 3}),
    }), {code: 'INCOMPLETE_BODY'});
});

test('aborting a stalled stream unblocks the read and destroys the body', async () => {
    const abort = new AbortController();
    let started;
    const waiting = new Promise(resolve => {started = resolve;});
    const body = new Readable({read: started});
    const read = readBoundedS3Object({
        key: 'part', maxBytes: 10, signal: abort.signal, getObject: async ({signal}) => {
            assert.equal(signal, abort.signal);
            return {Body: body};
        },
    });
    await waiting;
    abort.abort();
    await assert.rejects(read, /abort/i);
    assert.equal(body.destroyed, true);
});

test('conditionally moved manifest is distinct from transport failure', async () => {
    await assert.rejects(readBoundedS3Object({
        key: 'manifest', maxBytes: 100, ifMatch: '"old"', getObject: async ({ifMatch}) => {
            assert.equal(ifMatch, '"old"');
            throw Object.assign(new Error('precondition failed'), {$metadata: {httpStatusCode: 412}});
        },
    }), {code: 'MANIFEST_CHANGED'});
    const denied = Object.assign(new Error('access denied'), {$metadata: {httpStatusCode: 403}});
    await assert.rejects(readBoundedS3Object({
        key: 'manifest', maxBytes: 100, ifMatch: '"old"', getObject: async () => {throw denied;},
    }), error => error === denied);
});

test('oversized, canceled and highly compressed reads cannot replace an accepted report', async () => {
    let object;
    let now = new Date('2026-02-02T01:00:00Z');
    const manifest = JSON.parse(fixture('manifest.json'));
    const connector = createLocalCurConnector({readObject: () => object, now: () => now});
    const scope = {tenant: 'test', source: 'test', start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z'};
    const refresh = async (body, maxBytes = LIMITS.bytes, signal) => {
        const stage = await readBoundedS3Object({getObject: async () => ({Body: body}), key: manifest.reportKeys[0], maxBytes, signal});
        object = stage.bytes;
        return connector.refresh({...scope, manifest});
    };
    const accepted = await refresh(Readable.from([fixture('synthetic.csv.gz')]));
    assert.equal(accepted.report.TotalBilledCost, 100);
    now = new Date('2026-02-02T02:00:00Z');
    await assert.rejects(refresh(Readable.from([fixture('synthetic.csv.gz')]), 5), {code: 'OBJECT_TOO_LARGE'});
    assert.deepEqual(connector.get(scope), accepted);
    const abort = AbortSignal.abort();
    await assert.rejects(refresh(Readable.from([fixture('synthetic.csv.gz')]), LIMITS.bytes, abort), /abort/i);
    assert.deepEqual(connector.get(scope), accepted);
    const tooMuchExpanded = gzipSync(Buffer.alloc(LIMITS.bytes + 1, 'x'));
    await assert.rejects(refresh(Readable.from([tooMuchExpanded])), {code: 'ERR_BUFFER_TOO_LARGE'});
    assert.deepEqual(connector.get(scope), accepted);
});
