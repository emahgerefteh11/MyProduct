/* Source selection and stale-response fencing for a future private dashboard.
   No auth provider, storage implementation, or network endpoint is assumed.
   The connected read callback must use the verified server session. */
(function (root) {
    'use strict';

    function createReportViewSession({readConnected, readUpload, render}) {
        let generation = 0;
        let controller = null;
        function clear() {
            const ticket = ++generation;
            controller?.abort();
            controller = null;
            render({status: 'hidden'});
            return ticket;
        }
        async function open(source, selection) {
            const ticket = clear();
            const request = new AbortController();
            controller = request;
            render({source, status: 'loading'});
            try {
                const report = await (source === 'connected'
                    ? readConnected(selection, request.signal)
                    : readUpload());
                if (ticket !== generation) return;
                render(report == null
                    ? {source, status: 'unavailable'}
                    : {source, status: 'ready', report});
            } catch {
                // Avoid exposing provider messages or private data in view errors.
                if (ticket === generation) render({source, status: 'unavailable'});
            } finally {
                if (ticket === generation) controller = null;
            }
        }
        return {
            openConnected: selection => open('connected', selection),
            openUpload: () => open('upload'),
            // Call on sign-out, account switch, and before leaving a private page.
            // A restored page must keep content hidden until a fresh session read.
            clear,
        };
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {createReportViewSession};
    else root.createReportViewSession = createReportViewSession;
})(globalThis);
