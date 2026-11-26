'use strict';

const assert = require('node:assert/strict');
const {webcrypto} = require('node:crypto');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const {gzipSync} = require('node:zlib');
const {createLocalCurConnector} = require('../local_cur_connector.cjs');

const fixture = name => readFileSync(join(__dirname, 'fixtures', 'cur', name));
const code = name => readFileSync(join(__dirname, '..', name), 'utf8');
const manifest = JSON.parse(fixture('manifest.json'));
const raw = fixture('synthetic.csv');
const gzip = fixture('synthetic.csv.gz');
const tenant = 'fixture-tenant';
const source = 'fixture-source';
const period = {start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z'};
const scope = {tenant, source, ...period};

async function upload(bytes) {
    const worker = {postMessage: null};
    const context = vm.createContext({TextDecoder, crypto: webcrypto, self: worker});
    context.importScripts = file => {
        assert.equal(file, 'billing_import.js');
        vm.runInContext(code(file), context);
    };
    vm.runInContext(code('billing_worker.js'), context);
    async function run(data) {
        let response;
        worker.postMessage = message => { response = message; };
        await worker.onmessage({data: {id: 1, ...data}});
        assert.ok(response, 'the worker returned a response');
        assert.equal(response.error, undefined);
        return response.result;
    }
    const [file] = await run({type: 'prepare', delimiter: ',', files: [
        {name: 'synthetic.csv', size: bytes.length, arrayBuffer: async () => bytes},
    ]});
    const result = await run({type: 'validate', configs: [file.detected], currency: ''});
    assert.equal(result.ok, true, JSON.stringify(result.errors));
    return result.report;
}

function assertReport(report, total, unallocated) {
    assert.equal(report.sampleData, false);
    assert.equal(report.TotalBilledCost, total);
    const payments = report.products.find(p => p.Product === 'Payments');
    assert.equal(payments?.BilledCost, 70);
    assert.equal(payments.services.find(s => s.ServiceCategory === 'Object Storage')?.BilledCost, 70);
    assert.equal(report.allocation.unallocatedProductCost, unallocated);
    assert.equal(report.services.find(s => s.ServiceCategory === 'Compute/VM')?.BilledCost, unallocated);
    assert.equal(report.analysis.currency, 'USD');
    assert.equal(report.analysis.costBasis, 'billed');
    assert.equal(report.analysis.billingPeriod.start, period.start);
    assert.equal(report.analysis.billingPeriod.end, period.end);
    assert.equal(report.analysis.billingPeriod.complete, true);
    assert.equal(report.analysis.reconciliation.sourceTotal, total);
    assert.equal(report.analysis.reconciliation.difference, 0);
}

test('real upload and local manifest reconcile through retries, revisions and a failed refresh', async () => {
    let object = gzip;
    let time = new Date('2026-02-02T01:00:00Z');
    const connector = createLocalCurConnector({
        readObject: params => {
            assert.deepEqual(params, {tenant, source, key: manifest.reportKeys[0]});
            return object;
        },
        now: () => time,
    });

    const uploaded = await upload(raw);
    const initial = connector.refresh({tenant, source, manifest});
    assertReport(uploaded, 100, 30);
    assertReport(initial.report, 100, 30);
    assert.equal(uploaded.analysis.sources[0].sha256, initial.report.analysis.sources[0].sha256);
    assert.equal(initial.acceptedAt, time.toISOString());

    time = new Date('2026-02-02T02:00:00Z');
    assert.deepEqual(connector.refresh({tenant, source, manifest}), initial);
    assertReport(connector.get(scope).report, 100, 30);

    const corrected = Buffer.from(raw.toString().replace(',30,USD,AmazonEC2,', ',35,USD,AmazonEC2,'));
    object = gzipSync(corrected);
    const revised = connector.refresh({tenant, source, manifest});
    assertReport(await upload(corrected), 105, 35);
    assertReport(revised.report, 105, 35);
    assert.notEqual(revised.revision, initial.revision);
    assert.equal(revised.acceptedAt, time.toISOString());

    time = new Date('2026-02-02T03:00:00Z');
    object = Buffer.from('not gzip');
    assert.throws(() => connector.refresh({tenant, source, manifest}), /header check/);
    assert.deepEqual(connector.get(scope), revised);

    object = gzipSync(corrected.toString().replace(',35,USD,', ',oops,USD,'));
    assert.throws(() => connector.refresh({tenant, source, manifest}), /finite numeric amount/);
    assert.deepEqual(connector.get(scope), revised);
    assert.equal(connector.get({...scope, tenant: 'another-tenant'}), null);
});
