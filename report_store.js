/* Current report and opt-in invoice summaries, stored only in this browser. */
(function (root) {
    'use strict';
    const DATABASE = 'cloud-cost-atlas', STORE = 'reports', HISTORY = 'gcp-invoice-history';
    const MAX_MONTHS = 120;
    const treatments = Object.freeze({
        net_with_credits: 'After credits',
        before_credits: 'Before credits',
        unknown: 'Unknown credit treatment',
    });
    const hasTreatment = value => value === 'net_with_credits' || value === 'before_credits';
    const declaredScope = value => typeof value === 'string' && value.trim().length > 0 && value.trim().length <= 80;
    const slot = snapshot => JSON.stringify([snapshot.series, snapshot.month]);
    const context = snapshot => JSON.stringify([snapshot.series, snapshot.currency, snapshot.basis, snapshot.credits]);
    function snapshotFor(report, series, declaration) {
        const a = report?.analysis;
        if (report?.sampleData !== false || !/^[1-9]\d{3}(?:0[1-9]|1[0-2])$/.test(a?.invoiceMonth)) {
            throw new Error('Save history from one explicit Google Cloud invoice month at a time.');
        }
        if (!/^[A-Z]{3}$/.test(a.currency) || a.sources?.some(source => source.assumedCurrencyRows)) {
            throw new Error('The billing currency must be present on every row to save a month.');
        }
        if (!declaredScope(series)) throw new Error('Name the billing account or export scope using 1 to 80 characters.');
        if (!['billed', 'effective'].includes(a.costBasis) || !/^-?\d{1,16}(?:\.\d{1,4})?$/.test(a.invoiceTotal)) {
            throw new Error('This report does not contain a supported invoice total.');
        }
        const credits = hasTreatment(a.gcpCreditTreatment) ? a.gcpCreditTreatment
            : a.gcpCreditTreatment === 'not_declared' && hasTreatment(declaration) ? declaration : 'unknown';
        return {series: series.trim(), month: a.invoiceMonth, currency: a.currency, basis: a.costBasis,
            credits, total: a.invoiceTotal};
    }
    function compare(earlier, later) {
        if (!earlier || !later || earlier.month >= later.month || !hasTreatment(earlier.credits)
            || context(earlier) !== context(later)) return null;
        const places = Math.max(...[earlier, later].map(r => r.total.split('.')[1]?.length || 0));
        const scale = raw => {
            const [whole, fraction = ''] = raw.replace(/^-/, '').split('.');
            return (raw.startsWith('-') ? -1n : 1n) * BigInt(whole + fraction.padEnd(places, '0'));
        };
        const delta = scale(later.total) - scale(earlier.total);
        const abs = (delta < 0n ? -delta : delta).toString().padStart(places + 1, '0');
        return (delta < 0n ? '-' : '') + (places ? abs.slice(0, -places) + '.' + abs.slice(-places) : abs);
    }
    async function access(mode, operation) {
        if (!root.indexedDB) throw new Error('Browser storage is unavailable. Download the report or enable site storage to open it here.');
        const db = await new Promise((resolve, reject) => {
            const request = root.indexedDB.open(DATABASE, 1);
            request.onupgradeneeded = () => request.result.createObjectStore(STORE);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(new Error('Could not open browser report storage. Download the report or enable site storage.'));
            request.onblocked = () => reject(new Error('Close other Cloud Cost Atlas tabs, then try again.'));
        });
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE, mode);
            let result = () => null;
            transaction.oncomplete = () => { db.close(); resolve(result()); };
            transaction.onabort = transaction.onerror = () => { db.close(); reject(new Error('Could not save or read this report. Browser storage may be full or disabled. Download the report to keep a copy.')); };
            const request = operation(transaction.objectStore(STORE));
            result = typeof request === 'function' ? request : () => request.result ?? null;
        });
    }
    function updateHistory(action) {
        return access('readwrite', store => {
            const request = store.get(HISTORY);
            let result;
            request.onsuccess = () => {
                const current = request.result || [];
                const {next, outcome} = action(current);
                if (next) store.put(next, HISTORY);
                result = {outcome, summaries: next || current};
            };
            return () => result;
        });
    }
    root.ReportStore = {
        get: () => access('readonly', store => store.get('current')),
        save: report => access('readwrite', store => store.put(report, 'current')),
        clear: () => access('readwrite', store => store.delete('current')),
        getHistory: async () => await access('readonly', store => store.get(HISTORY)) || [],
        saveMonth: (report, series, declaration, confirmedId) => {
            const snapshot = snapshotFor(report, series, declaration);
            const id = root.crypto.randomUUID();
            return updateHistory(summaries => {
                const existing = summaries.find(s => slot(s) === slot(snapshot));
                if (existing && context(existing) !== context(snapshot)) return {outcome: 'incompatible'};
                if (existing?.total === snapshot.total) return {outcome: 'unchanged'};
                if (existing && existing.id !== confirmedId) return {outcome: 'confirm'};
                if (!existing && confirmedId) return {outcome: 'changed'};
                if (!existing && summaries.length >= MAX_MONTHS) return {outcome: 'full'};
                return {outcome: existing ? 'replaced' : 'saved', next: [...summaries.filter(s => s !== existing), {...snapshot, id}]};
            });
        },
        deleteMonth: id => updateHistory(summaries => {
            const next = summaries.filter(s => s.id !== id);
            return next.length === summaries.length ? {outcome: 'changed'} : {outcome: 'removed', next};
        }),
    };
    const history = Object.freeze({snapshotFor, compare, treatments, slot, declaredScope});
    root.InvoiceHistory = history;
    if (typeof module !== 'undefined' && module.exports) module.exports = history;
})(globalThis);
