'use strict';

const {LIMITS, sumDecimalAmounts} = require('./billing_import.js');
const HEADER = 'invoice.month,Cost,currency,service.description,cost_type,usage_start_time,usage_end_time,AtlasGcpCostBasis,Product';
const quote = value => '"' + String(value).replaceAll('"', '""') + '"';
const utf8Length = value => Buffer.byteLength(value, 'utf8');

function text(value) {
    if (value == null) return '';
    if (typeof value !== 'string' || value.length > 4096 || /[\u0000\r\n]/.test(value)) {
        throw new Error('Expected a bounded single-line GCP export field.');
    }
    return value;
}

function monetary(value) {
    if (!['number', 'string'].includes(typeof value) || value === '') {
        throw new Error('Missing or invalid cost or credit amount in GCP export.');
    }
    return value;
}

// Credential-free reference projection, not a BigQuery client. Input must be
// one complete decoded Standard export snapshot for the specified account/month.
// Do not silently filter an unexpected account/month out of the quoted total.
function projectGcpStandardRows(rows, {billingAccount, invoiceMonth} = {}) {
    if (typeof billingAccount !== 'string' || !billingAccount.trim()
        || typeof invoiceMonth !== 'string' || !/^[1-9]\d{3}(?:0[1-9]|1[0-2])$/.test(invoiceMonth)) {
        throw new Error('A billing account and invoice month (YYYYMM) are required.');
    }
    if (!Array.isArray(rows) || !rows.length || rows.length > LIMITS.rows) {
        throw new Error('Expected one nonempty bounded GCP export snapshot.');
    }
    let bytes = utf8Length(HEADER) + 1, currency;
    const lines = rows.map(row => {
        if (row?.billing_account_id !== billingAccount || row?.invoice?.month !== invoiceMonth) {
            throw new Error('A GCP row is outside the selected billing account or invoice month.');
        }
        if (typeof row.currency !== 'string' || !/^[A-Z]{3}$/.test(row.currency) || (currency && row.currency !== currency)) {
            throw new Error('The GCP invoice must have one explicit billing currency.');
        }
        currency = row.currency;
        // Empty repeated credits are valid. Missing or malformed fields are not
        // assumed credit-free; that would convert a partial response into a bill.
        if (!Array.isArray(row.credits) || row.credits.length > 128) throw new Error('Expected the GCP credits array.');
        const total = sumDecimalAmounts([monetary(row.cost), ...Array.from(row.credits, credit => monetary(credit?.amount))]);
        const line = [invoiceMonth, total, currency, text(row.service?.description), text(row.cost_type),
            text(row.usage_start_time), text(row.usage_end_time), 'net_with_credits', text(row.atlasProduct)]
            .map(quote).join(',');
        bytes += utf8Length(line) + 1;
        if (bytes > LIMITS.bytes) throw new Error('The projected GCP export exceeds the import size limit.');
        return line;
    });
    // Stable across reader ordering; repeated legitimate identical charge rows
    // stay separate. This is not a completeness proof or source revision.
    return Buffer.from(HEADER + '\n' + lines.sort().join('\n') + '\n');
}

module.exports = {projectGcpStandardRows};
