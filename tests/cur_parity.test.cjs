'use strict';

const assert = require('node:assert/strict');
const {webcrypto} = require('node:crypto');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Readable} = require('node:stream');
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

async function upload(bytes, costColumn = null, productColumn = null) {
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
        if (response.error) throw new Error(response.error);
        return response.result;
    }
    const parts = Array.isArray(bytes) ? bytes : [bytes];
    const files = await run({type: 'prepare', delimiter: ',', files: parts.map((part, i) => (
        {name: parts.length === 1 ? 'synthetic.csv' : `upload-${i + 1}.csv`, size: part.length, arrayBuffer: async () => part}
    ))});
    const configs = files.map(({detected}) => {
        if (costColumn) detected.mapping.cost = costColumn;
        if (productColumn) detected.mapping.product = productColumn;
        return detected;
    });
    const result = await run({type: 'validate', configs, currency: ''});
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

test('upload worker and CUR connector agree on three-decimal monetary amounts', async () => {
    const bytes = Buffer.from(raw.toString().replaceAll(',USD,', ',KWD,').replace(',30,KWD,', ',30.0024,KWD,'));
    const connector = createLocalCurConnector({readObject: () => gzipSync(bytes)});
    const connected = connector.refresh({tenant, source, manifest}).report;
    const uploaded = await upload(bytes);
    for (const report of [connected, uploaded]) {
        assert.equal(report.analysis.currency, 'KWD');
        assert.equal(report.TotalBilledCost, 100.002);
        assert.equal(report.allocation.unallocatedProductCost, 30.002);
        assert.equal(report.analysis.reconciliation.difference, -0.0004);
    }
});

test('part order is immaterial and incomplete or incompatible refreshes cannot publish', async () => {
    const [header, ...rows] = raw.toString().trimEnd().split('\n');
    const part = (...lines) => gzipSync([header, ...lines, ''].join('\n'));
    const [keyA, keyB] = manifest.reportKeys.concat('synthetic-cur/20260101-20260201/synthetic-00002.csv.gz');
    const keys = [keyA, keyB];
    const objects = new Map([[keyA, part(...rows.slice(0, 2))], [keyB, part(rows[2])]]);
    const manifestWith = reportKeys => ({...manifest, reportKeys});
    const connector = createLocalCurConnector({
        readObject: ({key}) => objects.get(key),
        now: () => new Date('2026-02-02T04:00:00Z'),
    });
    const refresh = (reportKeys = keys) => connector.refresh({tenant, source, manifest: manifestWith(reportKeys)});
    const initial = refresh();
    assertReport(initial.report, 100, 30);
    assert.equal(initial.report.analysis.sourceCount, 2);
    assert.deepEqual(refresh([...keys].reverse()), initial);

    objects.delete(keyB);
    assert.throws(() => refresh(), /object is missing/);
    assert.deepEqual(connector.get(scope), initial);

    objects.set(keyB, part(rows[2].replace(',USD,', ',EUR,')));
    assert.throws(() => refresh(), /Cannot combine USD, EUR/);
    assert.deepEqual(connector.get(scope), initial);

    objects.set(keyB, part(rows[2].replace('2026-01-01,2026-02-01', '2025-12-01,2026-01-01')));
    assert.throws(() => refresh(), /match the manifest billing period/);
    assert.deepEqual(connector.get(scope), initial);

    objects.set(keyB, gzipSync('EffectiveCost,BillingCurrency\n30,USD\n'));
    assert.throws(() => refresh(), /classic AWS CUR CSV/);
    assert.throws(() => refresh([keyA, 'part-00002.parquet']), /gzip CSV/);
    assert.throws(() => refresh([keyA, keyA]), /unique gzip CSV/);
    assert.deepEqual(connector.get(scope), initial);

    objects.set(keyB, part(rows[2].replace(',30,USD,', ',35,USD,')));
    const revised = refresh();
    assertReport(revised.report, 105, 35);
    assertReport(await upload(Buffer.from(raw.toString().replace(',30,USD,', ',35,USD,'))), 105, 35);
    assert.equal(connector.get({...scope, source: 'different-source'}), null);
    assert.equal(connector.get({...scope, end: '2026-03-01T00:00:00.000Z'}), null);
});

test('a kept CUR assembly cannot be rolled back or published after its pointer advances', () => {
    const dir = 'billing/synthetic-cur/20260101-20260201';
    const manifestKey = dir + '/synthetic-cur-Manifest.json';
    const version = assemblyId => ({...manifest, assemblyId, reportKeys: [dir + '/' + assemblyId + '/synthetic-cur-00001.csv.gz']});
    const original = version('20260202T010000Z');
    const corrected = version('20260202T020000Z');
    const later = version('20260202T030000Z');
    const updated = gzipSync(raw.toString().replace(',30,USD,AmazonEC2,', ',35,USD,AmazonEC2,'));
    const objects = new Map([[original.reportKeys[0], gzip], [corrected.reportKeys[0], updated], [later.reportKeys[0], updated]]);
    let current = original;
    let advanceDuringRead = false;
    let time = new Date('2026-02-02T01:05:00Z');
    const connector = createLocalCurConnector({
        readManifest: reference => {
            assert.deepEqual(reference, {tenant, source, key: manifestKey});
            return current;
        },
        readObject: ({key}) => {
            if (advanceDuringRead) current = later;
            return objects.get(key);
        },
        now: () => time,
    });
    const refresh = () => connector.refreshLatest({tenant, source, manifestKey});
    const first = refresh();
    assertReport(first.report, 100, 30);
    assert.equal(first.assemblyId, original.assemblyId);
    current = corrected;
    time = new Date('2026-02-02T02:05:00Z');
    advanceDuringRead = true;
    assert.throws(refresh, /advanced during refresh/);
    assert.deepEqual(connector.get(scope), first);
    advanceDuringRead = false;
    current = corrected;
    const accepted = refresh();
    assertReport(accepted.report, 105, 35);
    assert.equal(accepted.acceptedAt, time.toISOString());
    time = new Date('2026-02-02T03:05:00Z');
    current = original;
    assert.deepEqual(refresh(), accepted);
    assert.deepEqual(connector.get(scope), accepted);
    current = corrected;
    assert.deepEqual(refresh(), accepted);
    current = {...later, reportKeys: original.reportKeys};
    assert.throws(refresh, /outside its expected report assembly/);
    assert.deepEqual(connector.get(scope), accepted);
});

test('multipart pilot parity stays anchored to the accepted bytes after a new AWS delivery', async () => {
    const dir = 'billing/synthetic-cur/20260101-20260201';
    const manifestKey = dir + '/synthetic-cur-Manifest.json';
    const version = assemblyId => ({...manifest, assemblyId, reportKeys: [1, 2].map(i =>
        `${dir}/${assemblyId}/synthetic-cur-0000${i}.csv.gz`)});
    const [header, ...rows] = raw.toString().trimEnd().split('\n');
    const part = (...lines) => Buffer.from([header, ...lines, ''].join('\n'));
    const tagged = part(...rows.slice(0, 2)), untagged = part(rows[2]);
    const correctedUntagged = part(rows[2].replace(',30,USD,', ',35,USD,'));
    const first = version('20260202T010000Z'), next = version('20260202T020000Z');
    const objects = new Map([[first.reportKeys[0], gzipSync(tagged)], [first.reportKeys[1], gzipSync(untagged)],
        [next.reportKeys[0], gzipSync(tagged)], [next.reportKeys[1], gzipSync(correctedUntagged)]]);
    let pointer = first;
    const connector = createLocalCurConnector({readManifest: () => pointer, readObject: ({key}) => objects.get(key)});
    const refresh = () => connector.refreshLatest({tenant, source, manifestKey});
    const fingerprints = report => Array.from(report.analysis.sources, s => s.sha256).sort();

    const accepted = refresh();
    pointer = next;
    const manual = await upload([untagged, tagged]); // The accepted files, renamed and in reverse order.
    assert.deepEqual(fingerprints(manual), fingerprints(accepted.report));
    assertReport(manual, 100, 30);
    assertReport(accepted.report, 100, 30);
    const wrongRevision = await upload([correctedUntagged, tagged]);
    assert.notDeepEqual(fingerprints(wrongRevision), fingerprints(accepted.report));
    const corrected = refresh();
    assert.deepEqual(fingerprints(wrongRevision), fingerprints(corrected.report));
    assertReport(corrected.report, 105, 35);
});

test('a matching header-only CUR part adds no spend, but unavailable or invalid parts prevent publication', async () => {
    const dir = 'billing/synthetic-cur/20260101-20260201';
    const manifestKey = dir + '/synthetic-cur-Manifest.json';
    const version = assemblyId => ({...manifest, assemblyId, reportKeys: [1, 2].map(i =>
        dir + '/' + assemblyId + '/synthetic-cur-0000' + i + '.csv.gz')});
    const original = version('20260202T010000Z');
    const corrected = version('20260202T020000Z');
    const header = raw.toString().split('\n')[0] + '\n';
    const empty = gzipSync(header);
    const revisedRaw = Buffer.from(raw.toString().replace(',30,USD,AmazonEC2,', ',35,USD,AmazonEC2,'));
    const objects = new Map([[original.reportKeys[0], gzip], [original.reportKeys[1], empty]]);
    let current = original;
    let now = new Date('2026-02-02T01:05:00Z');
    const connector = createLocalCurConnector({readManifest: () => current, readObject: ({key}) => objects.get(key), now: () => now});
    const refresh = () => connector.refreshLatest({tenant, source, manifestKey});
    const accepted = refresh();
    const uploaded = await upload(raw);
    assertReport(uploaded, 100, 30);
    await assert.rejects(() => upload(Buffer.from(header)), /header but no billing rows/);
    assertReport(accepted.report, 100, 30);
    assert.deepEqual(accepted.report.analysis.sources.map(s => s.sha256), Array.from(uploaded.analysis.sources, s => s.sha256));
    assert.deepEqual(refresh(), accepted);

    now = new Date('2026-02-02T02:05:00Z');
    current = corrected;
    const [part, emptyPart] = corrected.reportKeys;
    objects.set(part, gzipSync(revisedRaw));
    for (const [bad, error] of [
        [undefined, /object is missing/],
        [Buffer.from('not gzip'), /header check/],
        [gzipSync(''), /file is empty/],
        [gzipSync('"unclosed'), /no closing quote/],
        [gzipSync(header.replace('Product', 'Different')), /match the headers/],
        [gzipSync(header + 'not-a-billing-row\n'), /match the manifest billing period/],
    ]) {
        objects.set(emptyPart, bad);
        assert.throws(refresh, error);
        assert.deepEqual(connector.get(scope), accepted);
    }
    objects.set(part, empty);
    objects.set(emptyPart, empty);
    assert.throws(refresh, /No billing rows/);
    assert.deepEqual(connector.get(scope), accepted);

    objects.set(part, gzipSync(revisedRaw));
    const revised = refresh();
    assertReport(await upload(revisedRaw), 105, 35);
    assertReport(revised.report, 105, 35);
    assert.equal(revised.acceptedAt, now.toISOString());

    now = new Date('2026-02-02T02:10:00Z');
    objects.set(emptyPart, gzipSync(header + '\n'));
    assert.throws(refresh, /changed an accepted assembly/);
    assert.deepEqual(connector.get(scope), revised);
});

test('pilot parity is pinned to the exact cost column, not the billed label alone', async () => {
    const unblendedColumn = 'lineItem/UnblendedCost';
    const netColumn = 'lineItem/NetUnblendedCost';
    const [header, storage, credit, compute] = raw.toString().trimEnd().split('\n');
    const csv = Buffer.from([
        header + ',' + netColumn, storage + ',68', credit + ',-10', compute + ',30', '',
    ].join('\n'));
    const connector = createLocalCurConnector({readObject: () => gzipSync(csv)});
    const connected = connector.refresh({tenant, source, manifest}).report;
    const uploaded = await upload(csv);
    assertReport(connected, 100, 30);
    assertReport(uploaded, 100, 30);
    for (const report of [connected, uploaded]) {
        assert.equal(report.analysis.sources[0].costColumn, unblendedColumn);
    }

    const net = await upload(csv, netColumn);
    assert.equal(net.analysis.costBasis, 'billed');
    assert.equal(net.analysis.sources[0].costColumn, netColumn);
    assert.equal(net.TotalBilledCost, 88);
    assert.equal(net.products.find(p => p.Product === 'Payments').BilledCost, 58);
    assert.equal(net.allocation.unallocatedProductCost, 30);
});

test('an exact server-owned product alias matches the manual upload and can remap one unchanged assembly', async () => {
    const dir = 'billing/synthetic-cur/20260101-20260201';
    const assemblyId = '20260202T010000Z';
    const manifestKey = dir + '/synthetic-cur-Manifest.json';
    const dataKey = dir + '/' + assemblyId + '/synthetic-cur-00001.csv.gz';
    const kept = {...manifest, assemblyId, reportKeys: [dataKey]};
    const owner = 'resourceTags/user:Owner';
    const costCenter = 'resourceTags/user:CostCenter';
    const [header, storage, credit, compute] = raw.toString().trimEnd().split('\n');
    const csv = Buffer.from([header.replace('resourceTags/user:Product', owner) + ',' + costCenter,
        storage + ',Shared', credit + ',Shared', compute + ',Shared', ''].join('\n'));
    let config = {version: 1, productColumn: owner};
    let time = new Date('2026-02-02T01:05:00Z');
    let switchWhileReading = null;
    const connector = createLocalCurConnector({
        readSourceMapping: async request => {
            assert.deepEqual(request, {tenant, source});
            return config;
        },
        readStreamObject: async ({key}) => {
            if (key === manifestKey) return {Body: Readable.from([Buffer.from(JSON.stringify(kept))])};
            assert.equal(key, dataKey);
            if (switchWhileReading) config = switchWhileReading;
            return {Body: Readable.from([gzipSync(csv)])};
        },
        now: () => time,
    });
    const refresh = () => connector.refreshLatestAsync({tenant, source, manifestKey, productColumn: 'ignored'});
    const first = await refresh();
    assertReport(await upload(csv, null, owner), 100, 30);
    assertReport(first.report, 100, 30);
    assert.equal(first.mappingVersion, 1);
    assert.equal(first.report.analysis.sources[0].columnMapping.product, owner);
    assert.deepEqual(await refresh(), first);

    time = new Date('2026-02-02T02:05:00Z');
    config = {version: 2, productColumn: owner};
    switchWhileReading = {version: 3, productColumn: costCenter};
    await assert.rejects(refresh(), /mapping advanced during refresh/);
    assert.deepEqual(connector.get(scope), first);

    switchWhileReading = null;
    const updated = await refresh();
    const manual = await upload(csv, null, costCenter);
    assert.equal(updated.revision, first.revision, 'immutable input bytes');
    assert.equal(updated.mappingVersion, 3);
    assert.equal(updated.acceptedAt, time.toISOString());
    assert.equal(updated.report.TotalBilledCost, 100);
    assert.deepEqual(updated.report.products, structuredClone(manual.products));
    assert.equal(updated.report.allocation.unallocatedProductCost, 0);

    config = {version: 3, productColumn: owner};
    await assert.rejects(refresh(), /without a new version/);
    config = {version: 2, productColumn: owner};
    await assert.rejects(refresh(), /stale CUR ownership mapping/);
    config = {version: 4, productColumn: 'resourceTags/user:costcenter'};
    await assert.rejects(refresh(), /not present in every populated/);
    assert.deepEqual(connector.get(scope), updated);
});

test('GCP projected net costs agree through upload, retries, corrections and failed refreshes', async () => {
    const {createLocalGcpConnector} = require('../local_gcp_connector.cjs');
    const jan = {...scope, invoiceMonth: '202601'};
    const csv = (month, untagged) => Buffer.from(
        'invoice.month,Cost,currency,service.description,Product,Platform,AtlasGcpCostBasis\n' +
        `${month},80,USD,Cloud Storage,Payments,Core,net_with_credits\n${month},-10,USD,Cloud Storage,Payments,Core,net_with_credits\n` +
        `${month},${untagged},USD,Compute Engine,,Core,net_with_credits\n`);
    let input = {revision: 1, bytes: csv('202601', 30)};
    let time = new Date('2026-02-02T01:00:00Z');
    const connector = createLocalGcpConnector({
        now: () => time, readExport: async req => {
            assert.deepEqual(req, {tenant, source, invoiceMonth: '202601'});
            return input;
        },
    });
    const check = async (snapshot, total, untagged) => {
        const manual = await upload(input.bytes);
        for (const report of [manual, snapshot.report]) {
            assert.equal(report.TotalBilledCost, total);
            assert.equal(report.analysis.invoiceMonth, '202601');
            assert.equal(report.products.find(p => p.Product === 'Payments').BilledCost, 70);
            assert.equal(report.allocation.unallocatedProductCost, untagged);
        }
        assert.equal(manual.analysis.sources[0].sha256, snapshot.sha256);
    };
    const first = await connector.refresh(jan);
    await check(first, 100, 30);
    time = new Date('2026-02-03T01:00:00Z');
    assert.deepEqual(await connector.refresh(jan), first);
    input = {revision: 2, bytes: csv('202601', 35)};
    const corrected = await connector.refresh(jan);
    await check(corrected, 105, 35);
    assert.notEqual(corrected.acceptedAt, first.acceptedAt);
    input = {revision: 3, bytes: csv('202602', 50)};
    await assert.rejects(connector.refresh(jan), /selected invoice month/);
    input = {revision: 3, bytes: Buffer.from(csv('202601', 50).toString().replace('50,USD', '50,EUR'))};
    await assert.rejects(connector.refresh(jan), /Cannot combine/);
    input = {revision: 3, bytes: Buffer.from(csv('202601', 50).toString().replaceAll('net_with_credits', 'gross'))};
    await assert.rejects(connector.refresh(jan), /already include credits/);
    input = {revision: 2, bytes: csv('202601', 50)};
    await assert.rejects(connector.refresh(jan), /cannot change/);
    input = {revision: 1, bytes: csv('202601', 30)};
    assert.deepEqual(await connector.refresh(jan), corrected);
    assert.deepEqual(connector.get(jan), corrected);
    assert.equal(connector.get({...jan, invoiceMonth: '202602'}), null);
    assert.equal(connector.get({...jan, tenant: 'other-tenant'}), null);
});

test('GCP native-shaped rows preserve signed credits, billing scope and cost types through both import paths', async () => {
    const {projectGcpStandardRows} = require('../local_gcp_projection.cjs');
    const {createLocalGcpConnector} = require('../local_gcp_connector.cjs');
    const B = require('../billing_import.js');
    const billingAccount = 'synthetic-gcp-billing';
    const invoiceMonth = '202610';
    const gcpScope = {tenant, source, invoiceMonth};
    const gcpRow = (cost, extra = {}) => ({
        billing_account_id: billingAccount, invoice: {month: invoiceMonth}, currency: 'USD',
        service: {description: 'Compute Engine'}, cost, credits: [], cost_type: 'regular',
        usage_start_time: '2026-10-01T00:00:00Z', usage_end_time: '2026-10-02T00:00:00Z', ...extra,
    });
    const rows = [
        gcpRow('80.00', {credits: [{amount: '-6.25'}, {amount: '-3.75'}], service: {description: 'Cloud Storage'},
            atlasProduct: 'Payments', usage_start_time: '2026-09-29T00:00:00Z', usage_end_time: '2026-09-30T00:00:00Z'}),
        gcpRow('27.97'),
        gcpRow('2.05', {cost_type: 'tax'}),
        gcpRow('-0.01', {cost_type: 'adjustment'}),
        gcpRow('-0.01', {cost_type: 'rounding error'}),
    ];
    const project = input => projectGcpStandardRows(input, {billingAccount, invoiceMonth});
    let input = {revision: 1, rows};
    const connector = createLocalGcpConnector({readExport: async scope => {
        assert.deepEqual(scope, gcpScope);
        return {revision: input.revision, bytes: project(input.rows)};
    }});
    const firstBytes = project(rows);
    assert.deepEqual(firstBytes, project(rows.slice().reverse()));
    const parsed = B.parseCSV(firstBytes.toString());
    assert.equal(parsed.records.length, 5);
    assert.deepEqual(parsed.records.map(r => r.cells[4]).sort(), ['adjustment', 'regular', 'regular', 'rounding error', 'tax']);
    assert.equal(B.parseCSV(project([gcpRow('0.1', {credits: [{amount: '0.2'}]})]).toString()).records[0].cells[1], '0.3');
    assert.equal(B.parseCSV(project([rows[1], rows[1]]).toString()).records.length, 2);
    const check = async (expected, unallocated) => {
        const manual = await upload(project(input.rows));
        const connected = await connector.refresh(gcpScope);
        for (const report of [manual, connected.report]) {
            assert.equal(report.TotalBilledCost, expected);
            assert.equal(report.analysis.gcpCreditTreatment, 'net_with_credits');
            assert.equal(report.analysis.invoiceMonth, invoiceMonth);
            assert.equal(report.analysis.usagePeriod.start, '2026-09-29T00:00:00.000Z');
            assert.equal(report.products.find(p => p.Product === 'Payments').BilledCost, 70);
            assert.equal(report.allocation.unallocatedProductCost, unallocated);
        }
        assert.equal(manual.analysis.sources[0].sha256, connected.sha256);
        return connected;
    };
    const first = await check(100, 30);
    input.rows = rows.slice().reverse();
    assert.deepEqual(await connector.refresh(gcpScope), first);
    input = {revision: 2, rows: rows.map((r, i) => i === 1 ? {...r, cost: '32.97'} : r)};
    const corrected = await check(105, 35);
    for (const [bad, reason] of [
        [{billing_account_id: 'different-account'}, /outside the selected/],
        [{invoice: {month: '202609'}}, /outside the selected/],
        [{currency: 'EUR'}, /explicit billing currency/],
        [{credits: undefined}, /credits array/],
        [{credits: [{amount: null}]}, /cost or credit amount/],
    ]) {
        input = {revision: 3, rows: [...rows, {...rows[0], ...bad}]};
        await assert.rejects(connector.refresh(gcpScope), reason);
        assert.deepEqual(connector.get(gcpScope), corrected);
    }
});

test('GCP query rehearsal publishes only all pages of one successful job with a stable count', async () => {
    const {collectGcpQueryProjection} = require('../local_gcp_query_results.cjs');
    const {createLocalGcpConnector} = require('../local_gcp_connector.cjs');
    const jobReference = {projectId: 'fixture-query-project', jobId: 'fixture-job', location: 'US'};
    const job = {jobReference, status: {state: 'DONE'}};
    const billingAccount = 'synthetic-account', invoiceMonth = '202610';
    const gcpScope = {tenant, source, invoiceMonth};
    const row = cost => ({billing_account_id: billingAccount, invoice: {month: invoiceMonth}, currency: 'USD',
        service: {description: 'Compute Engine'}, cost, credits: []});
    const first = {jobReference, jobComplete: true, totalRows: '2', rows: [row('60')], pageToken: 'next'};
    const next = {jobReference, jobComplete: true, rows: [row('40')]};
    let pages = [first, next], activeJob = job, revision = 1;
    const connector = createLocalGcpConnector({readExport: async () => {
        const requests = [];
        const bytes = await collectGcpQueryProjection({job: activeJob, billingAccount, invoiceMonth,
            readPage: async params => {
                assert.deepEqual(params.jobReference, jobReference);
                requests.push(params.pageToken);
                if (requests.length === 1) assert.equal(params.pageToken, undefined);
                if (requests.length === 2) assert.equal(params.pageToken, 'next');
                return pages[requests.length - 1];
            },
        });
        return {revision, bytes};
    }});
    const accepted = await connector.refresh(gcpScope);
    assert.equal(accepted.report.TotalBilledCost, 100);
    revision = 2;
    for (const [badPages, reason] of [
        [[{...first, pageToken: ''}], /before all rows/],
        [[first, {...next, totalRows: '3'}], /stays constant/],
        [[first, {...next, jobReference: {...jobReference, location: 'EU'}}], /completed query/],
        [[first, {...next, jobComplete: false}], /completed query/],
        [[first, {...next, rows: [row('41'), row('2')]}], /row count/],
        [[first, {...next, pageToken: 'next'}], /continuation/],
        [[{...first, totalRows: '100001'}], /bounded GCP result row count/],
        [[{...first, totalRows: '0', rows: []}], /bounded GCP result row count/],
        [[first, undefined], /completed query/],
    ]) {
        pages = badPages;
        await assert.rejects(connector.refresh(gcpScope), reason);
        assert.deepEqual(connector.get(gcpScope), accepted);
    }
    pages = [first, {...next, rows: [row('45')]}];
    activeJob = {...job, status: {state: 'DONE', errorResult: {reason: 'accessDenied'}}};
    await assert.rejects(connector.refresh(gcpScope), /successful completed/);
    assert.deepEqual(connector.get(gcpScope), accepted);
    activeJob = job;
    const corrected = await connector.refresh(gcpScope);
    assert.equal(corrected.report.TotalBilledCost, 105);
    assert.equal(connector.get(gcpScope).revision, 2);
});
