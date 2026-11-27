'use strict';

const ID = /^[A-Za-z0-9_-]{1,128}$/;
const PERIOD = /^[1-9]\d{3}-(?:0[1-9]|1[0-2])$/;
const headers = {'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'};
const json = (body, status, extra = {}) => Response.json(body, {status, headers: {...headers, ...extra}});
const valid = (pattern, value) => typeof value === 'string' && pattern.test(value);
const notFound = () => json({error: 'Not found.'}, 404);
const refused = () => json({error: 'This action is not allowed.'}, 403);
const unavailable = () => json({error: 'Private billing is unavailable. Try again.'}, 503);
const conflict = () => json({error: 'This action could not be applied. Reload and try again.'}, 409);

// A browser POST boundary only; it supplies no public route, AWS client or queue.
// The host supplies its canonical origin, verified current session, and a
// session-bound anti-CSRF check. Targets are parsed by the host; never use
// client-supplied tenant, AWS credentials, bucket, or role configuration.
function createPrivateBillingWrite({origin, resolveSession, verifyIntent, enqueueRefresh, disconnectSource}) {
    if (typeof origin !== 'string' || !/^https?:\/\//.test(origin) || new URL(origin).origin !== origin ||
        [resolveSession, verifyIntent, enqueueRefresh, disconnectSource].some(fn => typeof fn !== 'function')) {
        throw new TypeError('Canonical origin and private billing write dependencies are required.');
    }
    return async function handle(request, target) {
        if (request.method !== 'POST') return json({error: 'Method not allowed.'}, 405, {Allow: 'POST'});
        if (request.headers.get('Origin') !== origin) return refused();

        try {
            const session = await resolveSession(request);
            const tenant = session?.tenantId;
            if (!valid(ID, tenant)) return json({error: 'Sign-in required.'}, 401);
            if (session.canManageBilling !== true || await verifyIntent(request, session) !== true) return refused();
            if (!valid(ID, target?.source)) return notFound();
            const {source} = target;

            // Adapters recheck current grants and active connection in their
            // transaction. Refresh scopes and deduplicates requestId together
            // with the authorized job's immutable binding. No S3 read happens here.
            if (target.kind === 'refresh' && valid(PERIOD, target.period) && !target.period.startsWith('9999-')) {
                const requestId = request.headers.get('Idempotency-Key');
                if (!valid(ID, requestId)) return json({error: 'A valid request ID is required.'}, 400);
                const {period} = target;
                const job = await enqueueRefresh({tenant, source, period, requestId}, session);
                if (job == null) return notFound();
                if (job.status === 'conflict') return conflict();
                if (job.status === 'denied') return refused();
                if (job.status !== 'accepted' || job.tenant !== tenant || job.source !== source ||
                    job.period !== period || !valid(ID, job.jobId)) return unavailable();
                return json({status: 'accepted', jobId: job.jobId}, 202);
            }

            // Disconnect requires compare-and-swap on the generation selected by
            // the caller. A stale browser action cannot disconnect a new binding.
            if (target.kind === 'disconnect' && Number.isSafeInteger(target.generation) && target.generation > 0) {
                const result = await disconnectSource({tenant, source, generation: target.generation}, session);
                if (result == null) return notFound();
                if (result.status === 'conflict') return conflict();
                if (result.status === 'denied') return refused();
                if (result.status !== 'disconnected' || result.tenant !== tenant || result.source !== source) return unavailable();
                return json({status: 'disconnected'}, 200);
            }
            return notFound();
        } catch {
            return unavailable();
        }
    };
}

module.exports = {createPrivateBillingWrite};
