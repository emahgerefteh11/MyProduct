'use strict';

// An unmounted first-view prototype, not the accepted report or a storage/host
// choice. The caller must already have authorized access to the snapshot.
const MAX_OVERVIEW_BYTES = 32 * 1024;
const PREVIEW_LIMIT = 5;
const MAX_LABEL_JSON_BYTES = 384;
const DIMENSIONS = {platforms: 'Platform', products: 'Product', services: 'ServiceCategory',
    environments: 'Environment', providers: 'ProviderName'};

function finite(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Invalid billing overview amount.');
    return value;
}

function timestamp(value, nullable = false) {
    if (value === null && nullable) return null;
    if (typeof value !== 'string' || value.length > 35 || !Number.isFinite(Date.parse(value))) {
        throw new Error('Invalid billing overview timestamp.');
    }
    return value;
}

function coverage(value) {
    if (typeof value?.complete !== 'boolean') throw new Error('Invalid billing overview coverage.');
    return {start: timestamp(value.start, true), end: timestamp(value.end, true), complete: value.complete};
}

function groups(rows, key) {
    if (!Array.isArray(rows)) throw new Error('Invalid billing overview grouping.');
    const preview = [];
    // The importer already orders these groups by billed amount. Project only
    // its leading rows and don't shorten a name into a misleading identifier.
    for (const row of rows.slice(0, PREVIEW_LIMIT)) {
        const label = row?.[key];
        if (typeof label !== 'string') throw new Error('Invalid billing overview label.');
        if (label.length <= MAX_LABEL_JSON_BYTES && Buffer.byteLength(JSON.stringify(label)) <= MAX_LABEL_JSON_BYTES) {
            preview.push({label, amount: finite(row.BilledCost)});
        }
    }
    return {groupCount: rows.length, preview, incomplete: preview.length < rows.length};
}

function projectPrivateBillingOverview({report, acceptedAt}) {
    const analysis = report?.analysis;
    const currency = analysis?.currency;
    if (report?.sampleData !== false || (currency !== null && (typeof currency !== 'string' || !/^[A-Z]{3}$/.test(currency))) ||
        !['billed', 'effective'].includes(analysis?.costBasis) || !Number.isSafeInteger(analysis.rowCount) || analysis.rowCount < 1) {
        throw new Error('Invalid billing overview report.');
    }
    const overview = {
        kind: 'private-billing-overview',
        version: 1,
        totals: {amount: finite(report.TotalBilledCost), currency, costBasis: analysis.costBasis,
            sourceAmount: finite(analysis.reconciliation?.sourceTotal), rowCount: analysis.rowCount},
        allocation: {
            unallocatedProductCost: finite(report.allocation?.unallocatedProductCost),
            unallocatedPlatformCost: finite(report.allocation?.unallocatedPlatformCost),
            unallocatedEnvironmentCost: finite(report.allocation?.unallocatedEnvironmentCost),
        },
        billingPeriod: coverage(analysis.billingPeriod),
        usagePeriod: coverage(analysis.usagePeriod),
        lastSuccessfulSync: timestamp(acceptedAt),
        groups: Object.fromEntries(Object.entries(DIMENSIONS).map(([name, key]) => [name, groups(report[name], key)])),
    };
    if (Buffer.byteLength(JSON.stringify(overview)) > MAX_OVERVIEW_BYTES) throw new Error('Billing overview exceeded its budget.');
    return overview;
}

module.exports = {projectPrivateBillingOverview, MAX_OVERVIEW_BYTES};
