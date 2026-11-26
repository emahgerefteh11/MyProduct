'use strict';

// In-memory, credential-free adapter for classic CUR CSV manifests. The reader
// is scoped by its caller to one source; this is not a hosted AWS connector.
const {createHash} = require('node:crypto');
const {gunzipSync} = require('node:zlib');
const BillingImport = require('./billing_import.js');

function periodDate(value) {
    if (typeof value !== 'string' || !/^\d{8}T000000\.000Z$/.test(value)) return null;
    return BillingImport.dateOf(value.replace(/^(\d{4})(\d{2})(\d{2}).*$/, '$1-$2-$3'));
}

function createLocalCurConnector({readObject, now = () => new Date()}) {
    const accepted = new Map();
    const keyFor = ({tenant, source, start, end}) => JSON.stringify([tenant, source, start, end]);

    return {
        get(scope) {
            return structuredClone(accepted.get(keyFor(scope)) ?? null);
        },
        // This local API is synchronous so validation and replacement cannot
        // interleave. Only a complete validated snapshot is ever published.
        refresh({tenant, source, manifest}) {
            const start = periodDate(manifest?.billingPeriod?.start);
            const end = periodDate(manifest?.billingPeriod?.end);
            if (!tenant || !source || !start || !end || start >= end) {
                throw new Error('A tenant, source and valid CUR billing period are required.');
            }
            const keys = manifest.reportKeys;
            if (!Array.isArray(keys) || !keys.length || keys.length > BillingImport.LIMITS.files ||
                keys.some(key => typeof key !== 'string' || !key.endsWith('.csv.gz')) ||
                new Set(keys).size !== keys.length) {
                throw new Error('Expected unique gzip CSV report object keys within the import file limit.');
            }

            let bytesLeft = BillingImport.LIMITS.bytes;
            const files = [...keys].sort().map(key => {
                const compressed = readObject({tenant, source, key});
                if (!Buffer.isBuffer(compressed) || compressed.length > BillingImport.LIMITS.bytes || !bytesLeft) {
                    throw new Error('CUR object is missing or exceeds the import byte limit.');
                }
                const bytes = gunzipSync(compressed, {maxOutputLength: bytesLeft});
                bytesLeft -= bytes.length;
                const raw = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
                const parsed = BillingImport.parseCSV(raw, ',');
                return {name: key.split('/').at(-1).slice(0, -3), size: bytes.length, delimiter: ',', ...parsed,
                    sha256: createHash('sha256').update(bytes).digest('hex')};
            });
            const configs = files.map(file => BillingImport.detect(file.headers));
            if (configs.some(config => config.format !== 'aws')) throw new Error('Expected classic AWS CUR CSV columns.');
            for (const [i, file] of files.entries()) {
                const {billing_start, billing_end} = configs[i].mapping;
                const a = file.headers.indexOf(billing_start), b = file.headers.indexOf(billing_end);
                if (a < 0 || b < 0 || file.records.some(({cells}) =>
                    BillingImport.dateOf(cells[a]) !== start || BillingImport.dateOf(cells[b]) !== end)) {
                    throw new Error('CUR rows must match the manifest billing period.');
                }
            }
            const revision = createHash('sha256').update(JSON.stringify(files.map(file => file.sha256).sort())).digest('hex');
            const scope = {tenant, source, start, end};
            const key = keyFor(scope);
            const previous = accepted.get(key);
            if (previous?.revision === revision) return structuredClone(previous);

            const result = BillingImport.validate(files, configs, '');
            if (!result.ok) throw new Error(result.errors[0]?.message ?? 'CUR validation failed.');
            if (!result.report.analysis.currency) throw new Error('CUR currency must be specified.');
            const report = result.report;
            const acceptedAt = now().toISOString();
            Object.assign(report.analysis, {importMethod: 'local-cur-connector', generatedAt: acceptedAt});
            const snapshot = {revision, acceptedAt, report};
            accepted.set(key, snapshot);
            return structuredClone(snapshot);
        },
    };
}

module.exports = {createLocalCurConnector};
