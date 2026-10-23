/* Billing identities and an offline adoption package. No cloud credentials. */
(function (root) {
    'use strict';
    const MAX_BATCH = 200;
    const awsRegion = /^[a-z]{2}(?:-[a-z]+)+-\d+$/;
    function classify(record, defaults = {}) {
        const item = Object.fromEntries(['provider', 'account', 'region', 'zone', 'resourceId'].map(k => [k, String(record[k] || '').trim()]));
        Object.assign(item, {cost: Number(record.cost || 0), rows: Number(record.rows || 0), sourceHashes: Array.isArray(record.sourceHashes) ? record.sourceHashes : []});
        let rid = item.resourceId;
        const fail = message => ({...item, status: 'unresolved', message, resourceType: '', importId: ''});
        if (Object.values(item).some(v => typeof v === 'string' && (v.length > 2048 || /[\x00-\x1f]/.test(v)))) return fail('Invalid resource identifier.');
        if (!Number.isFinite(item.cost)) return fail('Resource cost must be finite.');
        if (item.provider === 'aws') {
            const arn = rid.match(/^arn:(aws|aws-us-gov|aws-cn):ec2:([a-z]{2}(?:-[a-z]+)+-\d+):(\d{12}):(instance|volume)\/((?:i|vol)-[0-9a-f]{8}(?:[0-9a-f]{9})?)$/);
            if (arn) {
                if ((arn[4] === 'instance') !== arn[5].startsWith('i-')) return fail('AWS ARN type conflicts with its resource ID.');
                if ((item.account && item.account !== arn[3]) || (item.region && item.region !== arn[2])) return fail('AWS ARN conflicts with the mapped account or region.');
                item.account = arn[3]; item.region = arn[2]; rid = arn[5];
            }
            item.account ||= defaults.awsAccount || ''; item.region ||= defaults.awsRegion || '';
            item.resourceType = /^i-[0-9a-f]{8}(?:[0-9a-f]{9})?$/.test(rid) ? 'aws_instance' : /^vol-[0-9a-f]{8}(?:[0-9a-f]{9})?$/.test(rid) ? 'aws_ebs_volume' : '';
            if (!item.resourceType) return fail('Unsupported type or descriptive name. Use a real EC2 instance or EBS volume ID.');
            if (!/^\d{12}$/.test(item.account) || !awsRegion.test(item.region)) return fail('Supply the usage account ID and region code to enable discovery.');
            item.importId = rid;
        } else if (item.provider === 'gcp') {
            rid = rid.replace(/^(?:\/\/compute\.googleapis\.com\/|https:\/\/www\.googleapis\.com\/compute\/v1\/)/, '');
            const match = rid.match(/^projects\/([a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{6,20})\/zones\/([a-z]+-[a-z]+\d+-[a-z])\/(instances|disks)\/([a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)$/);
            if (!match) return fail('Use a full projects/PROJECT/zones/ZONE/instances/NAME or disks/NAME resource path.');
            const region = match[2].replace(/-[a-z]$/, '');
            if ((item.account && item.account !== match[1]) || (item.zone && item.zone !== match[2]) || (item.region && item.region !== region)) return fail('GCP resource path conflicts with mapped scope.');
            item.account = match[1]; item.zone = match[2]; item.region = region;
            item.resourceType = match[3] === 'instances' ? 'google_compute_instance' : 'google_compute_disk'; item.importId = rid;
        } else return fail('Initial adoption supports AWS and GCP. Map the provider or keep this resource for manual review.');
        return {...item, status: 'candidate', message: 'Ready for live discovery. Configuration and unmanaged ownership still need verification.'};
    }
    function identity(item) { return JSON.stringify([item.provider, item.account, item.region, item.importId]); }
    function candidates(report, defaults = {}) {
        if (!report || !Array.isArray(report.resourceInventory)) throw new Error('This report has no scoped resource inventory. Reimport billing CSVs with the current importer or regenerate the report with the current Python analyzer.');
        if (report.resourceInventory.length > 100000) throw new Error('This report contains too many resources. Use a smaller billing period.');
        const normalized = report.resourceInventory.map(r => classify(r, defaults));
        const groups = new Map(), unresolved = [];
        for (const r of normalized) {
            if (r.status !== 'candidate') { unresolved.push(r); continue; }
            const key = identity(r);
            if (!groups.has(key)) groups.set(key, {...r, cost: 0, rows: 0, sourceHashes: []});
            const group = groups.get(key); group.cost += r.cost; group.rows += r.rows; group.sourceHashes.push(...r.sourceHashes);
            group.sourceHashes = [...new Set(group.sourceHashes)];
        }
        return [...groups.values(), ...unresolved].sort((a, b) => b.cost - a.cost);
    }
    function manifest(report, selected) {
        if (report.sampleData) throw new Error('Import your own billing report before preparing an adoption batch. Sample resources cannot be adopted.');
        if (!selected.length || selected.length > MAX_BATCH || selected.some(r => r.status !== 'candidate' || ['imported', 'imported-needs-review', 'already-managed'].includes(r.discovery?.status))) throw new Error(`Choose 1–${MAX_BATCH} unmanaged resources ready for discovery.`);
        if (new Set(selected.map(identity)).size !== selected.length) throw new Error('Select each cloud resource only once.');
        return {schemaVersion: 1, kind: 'atlas-adoption-manifest', generatedAt: new Date().toISOString(),
            billing: {generatedAt: report.analysis?.generatedAt || null, currency: report.analysis?.currency || null, costBasis: report.analysis?.costBasis || null},
            resources: selected.map(({provider, account, region, zone, resourceId, cost, sourceHashes}) => ({provider, account, region, zone, resourceId, cost, sourceHashes}))};
    }
    function results(inventory, items) {
        if (inventory?.schemaVersion !== 1 || inventory?.kind !== 'atlas-adoption-inventory' || !Array.isArray(inventory.resources) || inventory.resources.length > MAX_BATCH) throw new Error('Choose inventory.json produced by the Atlas adoption tool. Raw Terraform plan/state files are not accepted.');
        const allowed = new Set(['verified', 'imported', 'imported-needs-review', 'already-managed', 'not-found', 'error', 'dependency-review']);
        const map = new Map();
        for (const r of inventory.resources) {
            const canonical = classify(r);
            if (canonical.status !== 'candidate' || !allowed.has(r.status) || map.has(identity(canonical))) throw new Error('Discovery inventory has an invalid or duplicate resource identity/status.');
            // Ignore all arbitrary input attributes, including configuration.
            map.set(identity(canonical), {status: r.status, checkedAt: String(r.checkedAt || '').slice(0, 80), message: String(r.message || '').slice(0, 500)});
        }
        return items.map(item => ({...item, discovery: map.get(identity(item)) || null}));
    }

    // ZIP STORE format: small plain-text package, no dependency or CDN request.
    function zip(files) {
        const encoder = new TextEncoder(), local = [], central = [];
        let offset = 0;
        const bytes = (size, entries) => { const array = new Uint8Array(size), view = new DataView(array.buffer); for (const [at, value, width] of entries) width === 2 ? view.setUint16(at, value, true) : view.setUint32(at, value, true); return array; };
        const crc32 = data => { let crc = 0xffffffff; for (const byte of data) { crc ^= byte; for (let n = 0; n < 8; n++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1)); } return (crc ^ 0xffffffff) >>> 0; };
        for (const [path, content] of Object.entries(files)) {
            if (!/^[a-zA-Z0-9_.-]+$/.test(path)) throw new Error('Package filenames must be simple relative names.');
            const name = encoder.encode(path), data = encoder.encode(content), crc = crc32(data);
            const header = bytes(30, [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x800, 2], [14, crc, 4], [18, data.length, 4], [22, data.length, 4], [26, name.length, 2]]);
            local.push(header, name, data);
            central.push(bytes(46, [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x800, 2], [16, crc, 4], [20, data.length, 4], [24, data.length, 4], [28, name.length, 2], [42, offset, 4]]), name);
            offset += header.length + name.length + data.length;
        }
        const size = central.reduce((n, part) => n + part.length, 0), count = Object.keys(files).length;
        return new Blob([...local, ...central, bytes(22, [[0, 0x06054b50, 4], [8, count, 2], [10, count, 2], [12, size, 4], [16, offset, 4]])], {type: 'application/zip'});
    }
    const api = {MAX_BATCH, classify, candidates, manifest, results, identity, zip};
    if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Adoption = api;
})(globalThis);
