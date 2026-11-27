/* State for an explicitly selected private connected report. This helper does
   not install routes, start polling, or read/write the manual upload store. */
(function (root) {
    'use strict';

    const ID = /^[A-Za-z0-9_-]{1,128}$/;
    const validId = value => typeof value === 'string' && ID.test(value);
    const acceptedReport = value => value?.report?.sampleData === false &&
        typeof value.lastSuccessfulSync === 'string' && Number.isFinite(Date.parse(value.lastSuccessfulSync)) ? value : null;

    // A rejected write is not proof that report access was revoked. 403
    // covers read-only grants and intent failures. 409 can be an obsolete
    // generation. Recheck authoritative selection for either, without resending.
    async function decodeBillingActionResponse(response) {
        if (response.status === 401) return {status: 'denied'};
        if ([403, 404, 409].includes(response.status)) return {status: 'revalidate'};
        if (!response.ok) throw new Error('The billing action was not confirmed.');
        return response.json();
    }

    function createBillingRefreshView({submit, readJob, readReport, disconnect, readSelection, render,
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

        function useSelection(selection, accepted, canManageBilling, status) {
            const {source, period, generation} = selection;
            active = {scope: Object.freeze({source, period, generation}), accepted: acceptedReport(accepted),
                blocked: !canManageBilling};
            show(active, status);
        }

        function select(selection, accepted = null, {canManageBilling = false} = {}) {
            clear();
            // The caller supplies a verified selection and must select again
            // when the session changes. Do not send a tenant claim.
            useSelection(selection, accepted, canManageBilling === true,
                canManageBilling === true ? 'idle' : 'read-only');
        }

        async function revalidate(view, signal, stillCurrent) {
            view.blocked = true;
            view.accepted = null;
            show(view, 'verifying-access');
            try {
                signal.throwIfAborted();
                const next = await readSelection(view.scope, {signal});
                if (!stillCurrent()) return;
                if (next?.status === 'denied') return clear();
                if (!next?.scope || next.scope.source !== view.scope.source || next.scope.period !== view.scope.period ||
                    !Number.isSafeInteger(next.scope.generation) || next.scope.generation < 1 ||
                    (next.accepted != null && !acceptedReport(next.accepted))) {
                    throw new Error('Invalid billing selection.');
                }
                const canManage = next.canManageBilling === true;
                useSelection(next.scope, next.accepted, canManage, canManage ? 'selection-reloaded' : 'read-only');
            } catch {
                if (stillCurrent()) show(view, 'selection-unavailable');
            }
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
                if (result?.status === 'revalidate') return revalidate(view, signal, stillCurrent);
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
                if (result?.status === 'revalidate') return revalidate(view, signal, stillCurrent);
                if (result?.status !== 'disconnected') throw new Error('Unconfirmed disconnect.');
                view.requestId = view.jobId = null;
                show(view, 'disconnected');
            });
        }

        return {select, clear, refresh, check, stop};
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {createBillingRefreshView, decodeBillingActionResponse};
    else Object.assign(root, {createBillingRefreshView, decodeBillingActionResponse});
})(globalThis);
