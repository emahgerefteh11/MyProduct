/* Explicit, scoped private overview and drill-down. No host or routes are installed. */
(function (root) {
    'use strict';

    function createBillingDetailView({readOverview, readDetail, render}) {
        let view = null;
        function clear() {
            const previous = view;
            view = null;
            previous?.abort.abort();
            render({status: 'hidden', overview: null, detail: null});
        }

        function show(current, status) {
            render({status, overview: current.overview ?? null, detail: current.detail ?? null});
        }

        async function load(scope) {
            const previous = view;
            const current = {scope, abort: new AbortController()};
            view = current;
            previous?.abort.abort();
            show(current, 'loading');
            try {
                const result = await readOverview(scope, {signal: current.abort.signal});
                if (view !== current) return;
                if (result?.status === 'denied') return clear();
                if (typeof result?.acceptedRevision !== 'string' || !/^[a-f0-9]{64}$/.test(result.acceptedRevision) ||
                    result.overview?.kind !== 'private-billing-overview' || !Number.isFinite(result.overview?.totals?.amount)) {
                    throw new Error('Invalid overview.');
                }
                current.revision = result.acceptedRevision;
                current.overview = result.overview;
                show(current, 'ready');
            } catch {
                if (view === current) show(current, 'overview-unavailable');
            }
        }

        async function page(dimension, next = false) {
            const current = view;
            if (!current?.revision || current.pending) return;
            const offset = next && current.detail?.dimension === dimension ? current.detail.nextOffset : 0;
            if (!Number.isSafeInteger(offset) || offset < 0) return;
            current.pending = true;
            // Never show rows from another drill-down while the new one loads.
            current.detail = null;
            show(current, 'loading-detail');
            try {
                const result = await readDetail(current.scope, {dimension, offset, acceptedRevision: current.revision,
                    signal: current.abort.signal});
                if (view !== current) return;
                if (result?.status === 'denied') return clear();
                if (result?.status === 'stale') return load(current.scope);
                const detail = result?.detail;
                if (result?.acceptedRevision !== current.revision || detail?.dimension !== dimension ||
                    detail?.offset !== offset || !Array.isArray(detail.entries) ||
                    (detail.nextOffset !== null && (!Number.isSafeInteger(detail.nextOffset) || detail.nextOffset <= offset))) {
                    throw new Error('Invalid detail.');
                }
                current.detail = detail;
                show(current, 'detail-ready');
            } catch {
                if (view === current) show(current, 'detail-unavailable');
            } finally {
                current.pending = false;
            }
        }

        return {clear, load, page};
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {createBillingDetailView};
    else root.createBillingDetailView = createBillingDetailView;
})(globalThis);
