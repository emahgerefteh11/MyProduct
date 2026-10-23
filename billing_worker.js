'use strict';
importScripts('billing_import.js');
let prepared = [];
self.onmessage = async ({data}) => {
    const {id, type} = data;
    try {
        if (type === 'prepare') {
            prepared = [];
            const {files, delimiter} = data, {LIMITS} = BillingImport;
            if (!files.length || files.length > LIMITS.files) throw new Error(`Choose 1–${LIMITS.files} CSV files.`);
            if (files.reduce((n, f) => n + f.size, 0) > LIMITS.bytes) throw new Error('The selected files exceed 20 MiB combined. Use smaller exports or the command-line analyzer.');
            const next = [];
            for (const file of files) {
                try {
                    if (!/\.csv$/i.test(file.name)) throw new Error('Choose an uncompressed .csv file encoded as UTF-8.');
                    const bytes = await file.arrayBuffer();
                    const text = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
                    const parsed = BillingImport.parseCSV(text, delimiter);
                    const digest = await crypto.subtle.digest('SHA-256', bytes);
                    next.push({name: file.name, size: file.size, delimiter, ...parsed, sha256: Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')});
                } catch (e) { throw new Error(`${file.name}: ${e.message}`); }
            }
            if (next.reduce((n, f) => n + f.records.length, 0) > LIMITS.rows) throw new Error(`Choose at most ${LIMITS.rows.toLocaleString()} rows across all files.`);
            prepared = next;
            self.postMessage({id, result: next.map(f => ({name: f.name, size: f.size, headers: f.headers, rows: f.records.length, sample: f.records.slice(0, 5), detected: BillingImport.detect(f.headers)}))});
        } else if (type === 'validate') {
            if (!prepared.length) throw new Error('Choose and read your billing files first.');
            self.postMessage({id, result: BillingImport.validate(prepared, data.configs, data.currency)});
        } else throw new Error('Unknown import action.');
    } catch (e) { self.postMessage({id, error: e.message}); }
};
