'use strict';

// Synthetic local microbenchmark. No customer files, network, or cloud services.
// A fresh process per timing case excludes report generation from read-side RSS.
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const {mkdtempSync, rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {DatabaseSync} = require('node:sqlite');
const {performance} = require('node:perf_hooks');
const {parseCSV, detect, validate} = require('../billing_import.js');
const {createPrivateBillingRead} = require('../server/private_billing_read.cjs');

const tenant = 'synthetic', actor = 'reader', source = 'aws', period = '2026-01';
const acceptedAt = '2026-02-02T02:00:00.000Z';
const rows = 25_000, reads = 20;
const round = n => Number(n.toFixed(2));

function seed(file) {
    const db = new DatabaseSync(file);
    try {
        db.exec('CREATE TABLE snapshot (scenario TEXT PRIMARY KEY, tenant TEXT, actor TEXT, source TEXT, period TEXT, report TEXT)');
        const insert = db.prepare('INSERT INTO snapshot VALUES (?, ?, ?, ?, ?, ?)');
        for (const scenario of ['one-product', 'distinct-products']) {
            const lines = ['lineItem/UnblendedCost,lineItem/CurrencyCode,lineItem/ProductCode,lineItem/ResourceId,resourceTags/user:Product,resourceTags/user:Platform,bill/BillingPeriodStartDate,bill/BillingPeriodEndDate'];
            for (let i = 0; i < rows; i++) lines.push([
                i % 10 === 0 ? '-0.01' : '0.03', 'USD', 'AmazonEC2', 'benchmark-resource-' + i,
                scenario === 'one-product' ? 'benchmark-product' : 'benchmark-product-' + i,
                'benchmark-platform-' + i % 5, '2026-01-01', '2026-02-01',
            ].join(','));
            const parsed = parseCSV(lines.join('\n'));
            const result = validate([{name: 'synthetic.csv', sha256: 'synthetic', ...parsed}], [detect(parsed.headers)]);
            assert.equal(result.ok, true);
            assert.equal(result.report.TotalBilledCost, 650);
            insert.run(scenario, tenant, actor, source, period, JSON.stringify(result.report));
        }
    } finally {
        db.close();
    }
}

async function measure(file, scenario, batchSize) {
    const db = new DatabaseSync(file, {readOnly: true});
    try {
        const get = db.prepare('SELECT report FROM snapshot WHERE scenario=? AND tenant=? AND actor=? AND source=? AND period=?');
        const snapshotBytes = db.prepare('SELECT length(CAST(report AS BLOB)) AS bytes FROM snapshot WHERE scenario=?').get(scenario).bytes;
        const queryMs = [], parseMs = [], requestMs = [];
        const handle = createPrivateBillingRead({
            resolveSession: () => ({tenantId: tenant, actor}),
            acceptedTokenKey: Buffer.alloc(32, 31),
            readSelection(scope) {
                const start = performance.now();
                const row = get.get(scenario, scope.tenant, scope.actor, scope.source, scope.period);
                const fetched = performance.now();
                const report = JSON.parse(row.report);
                queryMs.push(fetched - start);
                parseMs.push(performance.now() - fetched);
                return {tenant, source, period, generation: 1, active: true, canManageBilling: false,
                    snapshot: {tenant, source, period, report, acceptedAt, assembly: 'synthetic', revision: 'one', mappingVersion: 1}};
            },
        });
        const rssBeforeMiB = round(process.memoryUsage.rss() / 1048576);
        let overviewBytes;
        const request = async () => {
            const start = performance.now();
            const result = await handle(new Request('http://127.0.0.1/overview'), {kind: 'overview', source, period});
            assert.equal(result.status, 200);
            const text = await result.text();
            const payload = JSON.parse(text);
            assert.equal(payload.overview.totals.amount, 650);
            assert.equal(payload.overview.totals.rowCount, rows);
            const size = Buffer.byteLength(text);
            if (overviewBytes !== undefined) assert.equal(size, overviewBytes);
            overviewBytes = size;
            requestMs.push(performance.now() - start);
        };
        for (let i = 0; i < reads; i += batchSize) await Promise.all(Array.from({length: batchSize}, request));
        const p95 = values => round(values.sort((a, b) => a - b)[Math.ceil(values.length * 0.95) - 1]);
        return {scenario, batchSize, reads, snapshotBytes, overviewBytes, rssBeforeMiB,
            maxRssMiB: round(process.resourceUsage().maxRSS / 1024),
            p95LocalReadMs: p95(requestMs), p95SqliteMs: p95(queryMs), p95ParseMs: p95(parseMs)};
    } finally {
        db.close();
    }
}

async function main() {
    if (process.argv[2] === '--measure') {
        const [, , , file, scenario, batch] = process.argv;
        assert.ok(file && ['one-product', 'distinct-products'].includes(scenario) && ['1', '4'].includes(batch),
            'Invalid internal benchmark arguments.');
        console.log(JSON.stringify(await measure(file, scenario, Number(batch))));
        return;
    }
    const dir = mkdtempSync(join(tmpdir(), 'atlas-overview-bench-'));
    try {
        const file = join(dir, 'synthetic.sqlite');
        seed(file);
        const results = [];
        for (const scenario of ['one-product', 'distinct-products']) {
            for (const batchSize of [1, 4]) {
                const worker = spawnSync(process.execPath, [__filename, '--measure', file, scenario, String(batchSize)],
                    {encoding: 'utf8', timeout: 60_000, maxBuffer: 1024 * 1024});
                if (worker.status !== 0) throw new Error(worker.stderr || String(worker.error));
                results.push(JSON.parse(worker.stdout));
            }
        }
        console.log(JSON.stringify({node: process.version, platform: process.platform, rows, results}, null, 2));
    } finally {
        rmSync(dir, {recursive: true, force: true});
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
