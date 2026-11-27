'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {parseCSV, detect, validate} = require('../billing_import.js');
const {projectPrivateBillingOverview, projectPrivateBillingDetail, MAX_OVERVIEW_BYTES} = require('../server/private_billing_overview.cjs');
const acceptedAt = '2026-02-02T02:00:00.000Z';

const csvCell = value => '"' + value.replaceAll('"', '""') + '"';

function reportFor(count, label) {
    const headers = ['lineItem/UnblendedCost','lineItem/CurrencyCode','lineItem/ProductCode','lineItem/ResourceId',
        'resourceTags/user:Product','resourceTags/user:Platform','bill/BillingPeriodStartDate','bill/BillingPeriodEndDate'];
    const lines = [headers.join(',')];
    for (let i = 0; i < count; i++) {
        lines.push([i % 10 === 0 ? '-0.01' : '0.03', 'USD', 'AmazonEC2',
            'private-resource-' + i, csvCell(label(i)),
            'platform-' + i % 5, '2026-01-01', '2026-02-01'].join(','));
    }
    const csv = lines.join('\n');
    const parsed = parseCSV(csv, ',');
    const config = detect(parsed.headers);
    const result = validate([{name: 'internal-name.csv', sha256: 'internal-checksum', ...parsed}], [config]);
    assert.equal(result.ok, true, JSON.stringify(result.errors));
    return result.report;
}

test('projection bounds first-view bytes for resource-heavy and high-cardinality products without changing totals', () => {
    for (const distinct of [false, true]) {
        const report = reportFor(25_000, i => distinct ? 'product-' + i : 'Product A');
        assert.equal(report.TotalBilledCost, 650);
        const projected = projectPrivateBillingOverview({report, acceptedAt});
        const fullBytes = Buffer.byteLength(JSON.stringify({report, lastSuccessfulSync: acceptedAt}));
        const reduced = {...report};
        delete reduced.resourceInventory;
        delete reduced.resourceNames;
        if (distinct) assert.ok(Buffer.byteLength(JSON.stringify(reduced)) > 10_000_000);
        const wire = JSON.stringify(projected);
        assert.ok(fullBytes > 5_000_000);
        assert.ok(Buffer.byteLength(wire) < MAX_OVERVIEW_BYTES);
        assert.doesNotMatch(wire, /internal-name|internal-checksum|private-resource|vmSkus/);
        assert.deepEqual(projected.totals, {amount: 650, sourceAmount: 650, costBasis: 'billed', currency: 'USD', rowCount: 25_000});
        assert.deepEqual(projected.allocation, report.allocation);
        assert.deepEqual(projected.billingPeriod, {start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z', complete: true});
        assert.equal(projected.lastSuccessfulSync, acceptedAt);
        assert.equal(projected.groups.products.groupCount, distinct ? 25_000 : 1);
        assert.equal(projected.groups.products.preview.length, distinct ? 5 : 1);
        assert.equal(projected.groups.products.incomplete, distinct);
        if (process.env.ATLAS_MEASURE_OVERVIEW === '1') {
            console.log({distinct, fullBytes, noResources: Buffer.byteLength(JSON.stringify(reduced)), projected: Buffer.byteLength(wire)});
        }
    }
});

test('projection does not truncate signed net, missing allocation, or oversized label identities', () => {
    const report = reportFor(10, i => i < 5 ? '' : '💸'.repeat(110));
    const actual = projectPrivateBillingOverview({report, acceptedAt});
    assert.equal(actual.totals.amount, 0.26);
    assert.equal(actual.allocation.unallocatedProductCost, 0.11);
    assert.equal(actual.groups.products.incomplete, true);
    assert.equal(actual.groups.products.groupCount, 2);
    assert.ok(actual.groups.products.preview.every(entry => !entry.label.includes('💸')));
    assert.throws(() => projectPrivateBillingOverview({report: {...report, sampleData: true}, acceptedAt}));
});

test('detail pages keep the canonical group order and do not return long raw labels', () => {
    const report = reportFor(9, i => i === 4 ? '💸'.repeat(110) : 'Product ' + i);
    const one = projectPrivateBillingDetail(report, 'products', 0);
    const two = projectPrivateBillingDetail(report, 'products', one.nextOffset);
    assert.equal(one.totalGroups, 9);
    assert.equal(one.nextOffset, 5);
    assert.equal(two.nextOffset, null);
    assert.deepEqual([...one.entries, ...two.entries], report.products.map(({Product, BilledCost}) =>
        ({label: Product.includes('💸') ? null : Product, amount: BilledCost})));
    assert.equal(projectPrivateBillingDetail(report, 'products', 10), null);
});
