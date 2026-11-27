/* State for an explicitly selected private connected report. This helper does
   not install routes, start polling, or read/write the manual upload store. */
(function (root) {
    'use strict';

    const ID = /^[A-Za-z0-9_-]{1,128}$/;
    const validId = value => typeof value === 'string' && ID.test(value);
    const acceptedReport = value => value?.report?.sampleData === false &&
        typeof value.lastSuccessfulSync === 'string' && Number.isFinite(Date.parse(value.lastSuccessfulSync)) ? value : null;

    function createBillingRefreshView({submit, readJob, readReport, disconnect, render,
        newRequestId = () => root.crypto.randomUUID()}) {
        let active = null;
        const current = (view, operation) => active === view && view.operation === operation;

        function show(view, status) {
            render({status, accepted: view.accepted});
        }

        function clear() {
            const previous = active;
            active = null;
            previous?.operation?.abort();
            render({status: 'hidden', accepted: null});
        }

        function select(selection, accepted = null) {
            clear();
            // The caller must select again (even with the same source/period)
            // when a session or tenant changes. Do not send a tenant claim.
            const {source, period, generation} = selection;
            active = {scope: Object.freeze({source, period, generation}), accepted: acceptedReport(accepted)};
            show(active, 'idle');
        }

        async function perform(view, onError, callback) {
            const operation = view.operation = new AbortController();
            const stillCurrent = () => current(view, operation);
            try {
                await callback(operation.signal, stillCurrent);
            } catch {
                if (stillCurrent()) show(view, onError);
            } finally {
                if (stillCurrent()) view.operation = null;
            }
        }

        async function refresh() {
            const view = active;
            if (!view || view.operation || view.jobId || view.blocked) return;
            await perform(view, 'retry-submit', async (signal, stillCurrent) => {
                show(view, 'submitting');
                signal.throwIfAborted();
                // Retain the same identity on an ambiguous submission. Only
                // an explicit subsequent action retries, never a timed loop.
                view.requestId ??= newRequestId();
                if (!validId(view.requestId)) throw new Error('Invalid refresh request ID.');
                const result = await submit(view.scope, {requestId: view.requestId, signal});
                if (!stillCurrent()) return;
                if (result?.status === 'denied') return clear();
                if (result?.status !== 'accepted' || !validId(result.jobId)) throw new Error('Invalid refresh confirmation.');
                view.jobId = result.jobId;
                show(view, 'queued');
            });
        }

        async function check() {
            const view = active;
            if (!view || view.operation || !view.jobId || view.blocked) return;
            await perform(view, 'retry-status', async (signal, stillCurrent) => {
                if (!view.readyToRead) {
                    const job = await readJob(view.jobId, {signal});
                    if (!stillCurrent()) return;
                    if (job?.status === 'denied') return clear();
                    if (job?.status === 'queued' || job?.status === 'running') return show(view, job.status);
                    if (job?.status === 'failed' || job?.status === 'canceled') {
                        view.requestId = view.jobId = null;
                        return show(view, job.status);
                    }
                    if (job?.status !== 'completed') throw new Error('Invalid refresh status.');
                    view.readyToRead = true;
                }
                show(view, 'reading-report');
                signal.throwIfAborted();
                const next = await readReport(view.scope, {signal});
                if (!stillCurrent()) return;
                if (next?.status === 'denied') return clear();
                const accepted = acceptedReport(next);
                if (!accepted) throw new Error('Latest accepted report is unavailable.');
                view.requestId = view.jobId = null;
                view.readyToRead = false;
                view.accepted = accepted;
                show(view, 'completed');
            });
        }

        async function stop() {
            const view = active;
            if (!view || view.blocked) return;
            view.blocked = true;
            const pending = view.operation;
            view.operation = null;
            pending?.abort();
            show(view, 'disconnecting');
            await perform(view, 'disconnect-unconfirmed', async (signal, stillCurrent) => {
                const result = await disconnect(view.scope, {signal});
                if (!stillCurrent()) return;
                if (result?.status === 'denied') return clear();
                if (result?.status !== 'disconnected') throw new Error('Unconfirmed disconnect.');
                view.requestId = view.jobId = null;
                show(view, 'disconnected');
            });
        }

        return {select, clear, refresh, check, stop};
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {createBillingRefreshView};
    else root.createBillingRefreshView = createBillingRefreshView;
})(globalThis);
