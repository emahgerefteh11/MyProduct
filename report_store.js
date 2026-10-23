/* A single accepted report, stored only in this browser's origin. */
(function (root) {
    'use strict';
    const DATABASE = 'cloud-cost-atlas', STORE = 'reports';
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
            const request = operation(transaction.objectStore(STORE));
            transaction.oncomplete = () => { db.close(); resolve(request.result ?? null); };
            transaction.onabort = transaction.onerror = () => { db.close(); reject(new Error('Could not save or read this report. Browser storage may be full or disabled. Download the report to keep a copy.')); };
        });
    }
    root.ReportStore = {
        get: () => access('readonly', store => store.get('current')),
        save: report => access('readwrite', store => store.put(report, 'current')),
        clear: () => access('readwrite', store => store.delete('current')),
    };
})(globalThis);
