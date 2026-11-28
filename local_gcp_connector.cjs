'use strict';

const {createHash} = require('node:crypto');
const billing = require('./billing_import.js');

// Credential-free prototype. The reader must return one complete, ordered
// revision of the documented net-cost CSV projection for this exact scope.
function createLocalGcpConnector({readExport, now = () => new Date()} = {}) {
    if (typeof readExport !== 'function') throw new Error('A scoped GCP export reader is required.');
    const accepted = new Map();
    function keyFor({tenant, source, invoiceMonth}) {
        if (typeof tenant !== 'string' || !tenant || typeof source !== 'string' || !source ||
            typeof invoiceMonth !== 'string' || !/^[1-9]\d{3}(?:0[1-9]|1[0-2])$/.test(invoiceMonth)) {
            throw new Error('A tenant, source and invoice month (YYYYMM) are required.');
        }
        return JSON.stringify([tenant, source, invoiceMonth]);
    }
    return {
        get(scope) {
            return structuredClone(accepted.get(keyFor(scope)) ?? null);
        },
        async refresh(scope) {
            const key = keyFor(scope);
            const {tenant, source, invoiceMonth} = scope;
            const {revision, bytes} = await readExport({tenant, source, invoiceMonth}) ?? {};
            if (!Number.isSafeInteger(revision) || revision < 1) throw new Error('Expected a positive source revision.');
            if (!Buffer.isBuffer(bytes) || !bytes.length || bytes.length > billing.LIMITS.bytes) {
                throw new Error('Expected a nonempty UTF-8 CSV export within the upload size limit.');
            }
            const before = accepted.get(key);
            if (before?.revision > revision) return structuredClone(before);
            const sha256 = createHash('sha256').update(bytes).digest('hex');
            if (before?.revision === revision) {
                if (before.sha256 !== sha256) throw new Error('A GCP revision cannot change its input bytes.');
                return structuredClone(before);
            }
            const file = {name: 'gcp-standard-net.csv', delimiter: ',', sha256,
                ...billing.parseCSV(new TextDecoder('utf-8', {fatal: true}).decode(bytes))};
            const fields = ['invoice.month', 'Cost', 'currency', 'service.description', 'AtlasGcpCostBasis'];
            if (fields.some(field => !file.headers.includes(field))) throw new Error('Expected the GCP Standard net-cost CSV projection.');
            const monthIndex = file.headers.indexOf('invoice.month'), currencyIndex = file.headers.indexOf('currency');
            if (file.records.some(row => row.cells[monthIndex]?.trim() !== invoiceMonth)) {
                throw new Error('Every GCP row must match the selected invoice month.');
            }
            if (file.records.some(row => !/^[A-Z]{3}$/.test(row.cells[currencyIndex]?.trim()))) {
                throw new Error('Every GCP row must specify its billing currency.');
            }
            const basisIndex = file.headers.indexOf('AtlasGcpCostBasis');
            if (file.records.some(row => row.cells[basisIndex]?.trim() !== 'net_with_credits')) {
                throw new Error('GCP projected costs must already include credits.');
            }
            const result = billing.validate([file], [billing.detect(file.headers, 'gcp')]);
            if (!result.ok) throw new Error(result.errors[0]?.message ?? 'GCP export validation failed.');
            if (result.report.analysis.invoiceMonth !== invoiceMonth) throw new Error('GCP invoice month could not be verified.');
            const acceptedAt = now().toISOString();
            Object.assign(result.report.analysis, {importMethod: 'local-gcp-connector', generatedAt: acceptedAt});
            const snapshot = {revision, sha256, acceptedAt, report: result.report};
            accepted.set(key, snapshot);
            return structuredClone(snapshot);
        },
    };
}
module.exports = {createLocalGcpConnector};
