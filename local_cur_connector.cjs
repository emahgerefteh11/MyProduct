'use strict';

// In-memory, credential-free adapter for classic CUR CSV manifests. Readers
// are scoped by their caller; this is not a hosted AWS connector.
const {createHash} = require('node:crypto');
const {gunzipSync} = require('node:zlib');
const BillingImport = require('./billing_import.js');
const {readBoundedS3Object} = require('./server/bounded_s3_read.cjs');
const MANIFEST_LIMIT = 256 * 1024;

function periodDate(value) {
    if (typeof value !== 'string' || !/^\d{8}T000000\.000Z$/.test(value)) return null;
    return BillingImport.dateOf(value.replace(/^(\d{4})(\d{2})(\d{2}).*$/, '$1-$2-$3'));
}

// The pilot supports kept versions with UTC timestamp assembly IDs. The
// period-level manifest is the only current pointer; do not list or guess keys.
function keptManifest(manifest, manifestKey) {
    const {assemblyId, reportName, billingPeriod, reportKeys} = manifest ?? {};
    const match = typeof assemblyId === 'string' && /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(assemblyId);
    const iso = match && `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}.000Z`;
    if (!iso || !Number.isFinite(Date.parse(iso)) || new Date(iso).toISOString() !== iso) {
        throw new Error('Expected a timestamped AWS CUR assembly ID (kept versions).');
    }
    const slash = manifestKey.lastIndexOf('/');
    const dir = manifestKey.slice(0, slash);
    const suffix = manifestKey.slice(slash + 1);
    const prefix = `${dir}/${assemblyId}/`;
    const start = periodDate(billingPeriod?.start), end = periodDate(billingPeriod?.end);
    if (!start || !end || start >= end || slash < 1 || suffix !== `${reportName}-Manifest.json` ||
        typeof reportName !== 'string' || !reportName || !dir.endsWith('/' + billingPeriod.start.slice(0, 8) + '-' + billingPeriod.end.slice(0, 8))) {
        throw new Error('CUR manifest must match the configured report and billing period.');
    }
    if (!Array.isArray(reportKeys) || reportKeys.some(key => typeof key !== 'string' ||
        !key.startsWith(prefix) || !key.slice(prefix.length).startsWith(reportName + '-') ||
        key.slice(prefix.length).includes('/'))) {
        throw new Error('CUR manifest contains a part outside its expected report assembly.');
    }
    const signature = JSON.stringify([reportName, start, end, assemblyId, [...reportKeys].sort()]);
    return {manifest: {assemblyId, reportName, billingPeriod: {...billingPeriod}, reportKeys: [...reportKeys]}, signature};
}

function partKeys(manifest) {
    const keys = manifest.reportKeys;
    if (!Array.isArray(keys) || !keys.length || keys.length > BillingImport.LIMITS.files ||
        keys.some(key => typeof key !== 'string' || !key.endsWith('.csv.gz')) ||
        new Set(keys).size !== keys.length) {
        throw new Error('Expected unique gzip CSV report object keys within the import file limit.');
    }
    return [...keys].sort();
}

function createLocalCurConnector({readObject, readManifest, readStreamObject, now = () => new Date()}) {
    const accepted = new Map();
    const keyFor = ({tenant, source, start, end}) => JSON.stringify([tenant, source, start, end]);

    // Final pointer verification happens after validating all parts and before
    // committing. A streamed refresh may pause for that verification.
    function refresh({tenant, source, manifest}, current = null) {
        const start = periodDate(manifest?.billingPeriod?.start);
        const end = periodDate(manifest?.billingPeriod?.end);
        if (!tenant || !source || !start || !end || start >= end) {
            throw new Error('A tenant, source and valid CUR billing period are required.');
        }
        const keys = partKeys(manifest);

        const scope = {tenant, source, start, end};
        const key = keyFor(scope);
        const previous = accepted.get(key);
        const assemblyId = current ? manifest.assemblyId : null;
        if (assemblyId && previous?.assemblyId > assemblyId) return structuredClone(previous);
        if (assemblyId && previous?.assemblyId === assemblyId && previous.manifestSignature !== current.signature) {
            throw new Error('CUR manifest changed an accepted assembly; keeping the previous report.');
        }

        let bytesLeft = BillingImport.LIMITS.bytes;
        const files = keys.map(key => {
            const compressed = current?.objects ? current.objects.get(key) : readObject({tenant, source, key});
            if (!Buffer.isBuffer(compressed) || compressed.length > BillingImport.LIMITS.bytes || !bytesLeft) {
                throw new Error('CUR object is missing or exceeds the import byte limit.');
            }
            const bytes = gunzipSync(compressed, {maxOutputLength: bytesLeft});
            bytesLeft -= bytes.length;
            const raw = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
            const parsed = BillingImport.parseCSV(raw, ',', {allowHeaderOnly: true});
            return {name: key.split('/').at(-1).slice(0, -3), size: bytes.length, delimiter: ',', ...parsed,
                sha256: createHash('sha256').update(bytes).digest('hex')};
        });
        const populated = files.filter(file => file.records.length);
        if (!populated.length) throw new Error('No billing rows were available in the CUR assembly.');
        // A listed empty partition must still be readable, valid gzip CSV, and
        // have the same headers as a populated part. Keep it in the revision
        // hash and limits, but not in the aggregation or period coverage.
        const populatedHeaders = new Set(populated.map(file => JSON.stringify(file.headers)));
        if (files.some(file => !file.records.length && !populatedHeaders.has(JSON.stringify(file.headers)))) {
            throw new Error('An empty CUR partition must match the headers of a populated part.');
        }
        const configs = populated.map(file => BillingImport.detect(file.headers));
        if (configs.some(config => config.format !== 'aws')) throw new Error('Expected classic AWS CUR CSV columns.');
        for (const [i, file] of populated.entries()) {
            const {billing_start, billing_end} = configs[i].mapping;
            const a = file.headers.indexOf(billing_start), b = file.headers.indexOf(billing_end);
            if (a < 0 || b < 0 || file.records.some(({cells}) =>
                BillingImport.dateOf(cells[a]) !== start || BillingImport.dateOf(cells[b]) !== end)) {
                throw new Error('CUR rows must match the manifest billing period.');
            }
        }
        const revision = createHash('sha256').update(JSON.stringify(files.map(file => file.sha256).sort())).digest('hex');
        if (assemblyId && previous?.assemblyId === assemblyId && previous.revision !== revision) {
            throw new Error('CUR data changed an accepted assembly; keeping the previous report.');
        }
        const identical = previous?.revision === revision;
        if (identical && !current) return structuredClone(previous);

        const result = BillingImport.validate(populated, configs, '');
        if (!result.ok) throw new Error(result.errors[0]?.message ?? 'CUR validation failed.');
        if (!result.report.analysis.currency) throw new Error('CUR currency must be specified.');
        function finish(latest) {
            current?.signal?.throwIfAborted();
            if (current && latest.signature !== current.signature) {
                throw new Error('CUR current manifest advanced during refresh; retry the new revision.');
            }
            // Re-read at commit time, not before an awaited manifest read.
            const last = accepted.get(key);
            if (assemblyId && last?.assemblyId > assemblyId) return structuredClone(last);
            if (current && last && last.assemblyId === assemblyId) {
                if (last.revision !== revision || last.manifestSignature !== current.signature) {
                    throw new Error('CUR data changed an accepted assembly; keeping the previous report.');
                }
                return structuredClone(last);
            }
            const report = result.report;
            const acceptedAt = now().toISOString();
            Object.assign(report.analysis, {importMethod: 'local-cur-connector', generatedAt: acceptedAt});
            const snapshot = {revision, acceptedAt, report};
            if (current) Object.assign(snapshot, {assemblyId, manifestSignature: current.signature});
            accepted.set(key, snapshot);
            return structuredClone(snapshot);
        }
        return current?.objects ? current.readAgain().then(finish) : finish(current?.readAgain());
    }

    return {
        get(scope) {
            return structuredClone(accepted.get(keyFor(scope)) ?? null);
        },
        refresh(input) {
            if (readManifest || readStreamObject) throw new Error('Versioned connectors must refresh from the current CUR manifest.');
            return refresh(input);
        },
        async refreshLatestAsync({tenant, source, manifestKey, signal}) {
            if (!readStreamObject || !tenant || !source || typeof manifestKey !== 'string' || !manifestKey) {
                throw new Error('A tenant, source, manifest key, and scoped stream reader are required.');
            }
            const read = (key, maxBytes, ifMatch) => readBoundedS3Object({
                getObject: input => readStreamObject({tenant, source, ...input}),
                key, maxBytes, ifMatch, signal,
            });
            const current = await read(manifestKey, MANIFEST_LIMIT);
            const validate = bytes => keptManifest(JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)), manifestKey);
            const {manifest, signature} = validate(current.bytes);
            const objects = new Map();
            let compressedLeft = BillingImport.LIMITS.bytes;
            for (const key of partKeys(manifest)) {
                if (!compressedLeft) throw new Error('CUR assembly exceeds the compressed byte limit.');
                const {bytes} = await read(key, compressedLeft);
                compressedLeft -= bytes.length;
                objects.set(key, bytes);
            }
            const readAgain = async () => validate((await read(manifestKey, MANIFEST_LIMIT, current.etag ?? undefined)).bytes);
            return refresh({tenant, source, manifest}, {signature, objects, signal, readAgain});
        },
        refreshLatest({tenant, source, manifestKey}) {
            if (!readManifest || !tenant || !source || typeof manifestKey !== 'string' || !manifestKey) {
                throw new Error('A tenant, source, manifest key, and current manifest reader are required.');
            }
            const readAgain = () => keptManifest(readManifest({tenant, source, key: manifestKey}), manifestKey);
            const {manifest, signature} = readAgain();
            return refresh({tenant, source, manifest}, {signature, readAgain});
        },
    };
}

module.exports = {createLocalCurConnector};
