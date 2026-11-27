'use strict';

const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Readable} = require('node:stream');
const test = require('node:test');
const {gzipSync} = require('node:zlib');
const {LIMITS} = require('../billing_import.js');
const {createLocalCurConnector} = require('../local_cur_connector.cjs');

const fixture = name => readFileSync(join(__dirname, 'fixtures', 'cur', name));
const base = JSON.parse(fixture('manifest.json'));
const [header, ...rows] = fixture('synthetic.csv').toString().trimEnd().split('\n');
const dir = 'billing/synthetic-cur/20260101-20260201';
const manifestKey = dir + '/synthetic-cur-Manifest.json';
const credentialsScope = {tenant: 'fixture-tenant', source: 'fixture-source'};
const reportScope = {...credentialsScope, start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z'};

function setup() {
    const version = assemblyId => ({...base, assemblyId, reportKeys: [1, 2].map(n =>
        dir + '/' + assemblyId + '/synthetic-cur-0000' + n + '.csv.gz')});
    const initial = version('20260202T010000Z');
    const revised = version('20260202T020000Z');
    const newest = version('20260202T030000Z');
    const part = (...lines) => gzipSync([header, ...lines, ''].join('\n'));
    const data = new Map();
    for (const [version, amount] of [[initial, 30], [revised, 35], [newest, 40]]) {
        data.set(version.reportKeys[0], part(...rows.slice(0, 2)));
        data.set(version.reportKeys[1], part(rows[2].replace(',30,USD,', ',' + amount + ',USD,')));
    }

    const h = {initial, revised, newest, data, current: initial, tick: 1, log: [], hook: null};
    h.connector = createLocalCurConnector({
        now: () => new Date(Date.UTC(2026, 1, 2, h.tick)),
        readStreamObject: async opts => {
            assert.equal(opts.tenant, credentialsScope.tenant);
            assert.equal(opts.source, credentialsScope.source);
            h.log.push([opts.key, opts.ifMatch]);
            const handled = await h.hook?.(opts);
            if (handled) return handled;
            if (opts.key === manifestKey) {
                const etag = '"' + h.current.assemblyId + '"';
                if (opts.ifMatch && opts.ifMatch !== etag) throw Object.assign(new Error('changed'), {$metadata: {httpStatusCode: 412}});
                return {Body: Readable.from([Buffer.from(JSON.stringify(h.current))]), ETag: etag};
            }
            const bytes = h.data.get(opts.key);
            if (!bytes) throw new Error('fixture part missing');
            return {Body: Readable.from([bytes])};
        },
    });
    h.refresh = signal => h.connector.refreshLatestAsync({...credentialsScope, manifestKey, signal});
    h.get = () => h.connector.get(reportScope);
    return h;
}

test('bounded multipart refresh stages parts, checks current manifest, and accepts corrections only once', async () => {
    const h = setup();
    const first = await h.refresh();
    assert.equal(first.report.TotalBilledCost, 100);
    assert.equal(first.report.analysis.sourceCount, 2);
    assert.equal(first.report.allocation.unallocatedProductCost, 30);
    assert.deepEqual(h.log, [
        [manifestKey, undefined],
        ...h.initial.reportKeys.map(key => [key, undefined]),
        [manifestKey, '"' + h.initial.assemblyId + '"'],
    ]);
    h.tick = 2;
    assert.deepEqual(await h.refresh(), first);
    h.current = h.revised;
    const corrected = await h.refresh();
    assert.equal(corrected.report.TotalBilledCost, 105);
    assert.equal(corrected.report.allocation.unallocatedProductCost, 35);
    assert.equal(corrected.acceptedAt, '2026-02-02T02:00:00.000Z');
    assert.notEqual(corrected.revision, first.revision);
});

test('moved, canceled, over-budget and out-of-scope reads cannot replace the accepted report', async () => {
    const h = setup();
    const first = await h.refresh();
    h.tick = 2;
    h.current = h.revised;
    h.hook = ({key}) => {if (key === h.revised.reportKeys[1]) h.current = h.newest;};
    await assert.rejects(h.refresh(), {code: 'MANIFEST_CHANGED'});
    assert.deepEqual(h.get(), first);

    h.current = h.revised;
    const cancel = new AbortController();
    const body = new Readable({read: () => cancel.abort()});
    h.hook = ({key}) => key === h.revised.reportKeys[1] ? {Body: body} : null;
    await assert.rejects(h.refresh(cancel.signal), /abort/i);
    assert.equal(body.destroyed, true);
    assert.deepEqual(h.get(), first);

    let oversized;
    h.hook = ({key}) => key === h.revised.reportKeys[1]
        ? {Body: oversized = Readable.from([]), ContentLength: LIMITS.bytes}
        : null;
    await assert.rejects(h.refresh(), {code: 'OBJECT_TOO_LARGE'});
    assert.equal(oversized.destroyed, true);
    assert.deepEqual(h.get(), first);

    h.current = {...h.revised, reportKeys: h.initial.reportKeys};
    h.hook = null;
    h.log.length = 0;
    await assert.rejects(h.refresh(), /outside its expected report assembly/);
    assert.deepEqual(h.log, [[manifestKey, undefined]]);
    assert.deepEqual(h.get(), first);
});

test('a later-finishing worker cannot replace a newer commit after its final pointer check', async () => {
    const h = setup();
    await h.refresh();
    let reachedFinal, releaseFinal;
    const reached = new Promise(resolve => {reachedFinal = resolve;});
    const release = new Promise(resolve => {releaseFinal = resolve;});
    h.hook = async ({key, ifMatch}) => {
        if (key !== manifestKey || ifMatch !== '"' + h.revised.assemblyId + '"') return null;
        reachedFinal();
        await release;
        return {Body: Readable.from([Buffer.from(JSON.stringify(h.revised))]), ETag: ifMatch};
    };
    h.current = h.revised;
    const finishingLate = h.refresh();
    await reached;
    try {
        h.tick = 3;
        h.current = h.newest;
        const accepted = await h.refresh();
        assert.equal(accepted.report.TotalBilledCost, 110);
        h.tick = 4;
        releaseFinal();
        assert.deepEqual(await finishingLate, accepted);
        assert.deepEqual(h.get(), accepted);
    } finally {
        releaseFinal();
        await finishingLate.catch(() => {});
    }
});
