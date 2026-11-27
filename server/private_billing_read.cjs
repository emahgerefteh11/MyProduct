'use strict';

// Server-only boundary. The host must supply a verified, current session and
// tenant-scoped storage, not a tenant claim taken directly from the request.
// This module installs no routes and contains no credentials or cloud client.
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const PERIOD = /^\d{4}-(?:0[1-9]|1[0-2])$/;
const JOB_STATES = new Set(['queued', 'running', 'completed', 'failed', 'canceled']);
const headers = {'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'};
const json = (body, status = 200, extra = {}) => Response.json(body, {status, headers: {...headers, ...extra}});
const notFound = () => json({error: 'Not found.'}, 404);
const unavailable = () => json({error: 'Private billing is unavailable. Try again.'}, 503);
const valid = (pattern, value) => typeof value === 'string' && pattern.test(value);

function reportEnvelope(snapshot) {
    const {report, acceptedAt} = snapshot;
    if (!report || report.sampleData !== false || typeof acceptedAt !== 'string' ||
        !Number.isFinite(Date.parse(acceptedAt))) throw new Error('Invalid accepted report.');
    return {report, lastSuccessfulSync: acceptedAt};
}

// readSelection must confirm the actor's current read grant and return the
// connection and optional accepted snapshot in one consistent scoped read.
// A missing grant/connection/period returns null. A disconnected source may
// retain a report for authorized readers but must never enable writes.
function createPrivateBillingRead({resolveSession, readReport, readJob, readSelection}) {
    return async function handle(request, target) {
        if (request.method !== 'GET') return json({error: 'Method not allowed.'}, 405, {Allow: 'GET'});

        try {
            const session = await resolveSession(request);
            const tenant = session?.tenantId;
            if (!valid(ID, tenant)) return json({error: 'Sign-in required.'}, 401);

            if (target?.kind === 'report' && valid(ID, target.source) && valid(PERIOD, target.period)) {
                const snapshot = await readReport({tenant, source: target.source, period: target.period});
                if (snapshot == null) return notFound();
                return json(reportEnvelope(snapshot));
            }
            if (target?.kind === 'selection' && valid(ID, target.source) && valid(PERIOD, target.period)) {
                if (!valid(ID, session.actor)) return json({error: 'Sign-in required.'}, 401);
                const {source, period} = target;
                const selection = await readSelection({tenant, actor: session.actor, source, period});
                if (selection == null) return notFound();
                const {generation, active, canManageBilling, snapshot} = selection;
                if (selection.tenant !== tenant || selection.source !== source || selection.period !== period ||
                    !Number.isSafeInteger(generation) || generation < 1 || typeof active !== 'boolean' ||
                    typeof canManageBilling !== 'boolean' || (snapshot != null &&
                    (snapshot.tenant !== tenant || snapshot.source !== source || snapshot.period !== period))) {
                    return unavailable();
                }
                return json({scope: {source, period, generation},
                    connectionStatus: active ? 'active' : 'disconnected',
                    canManageBilling: active && canManageBilling,
                    accepted: snapshot == null ? null : reportEnvelope(snapshot)});
            }
            if (target?.kind === 'job' && valid(ID, target.jobId)) {
                const job = await readJob({tenant, jobId: target.jobId});
                if (job == null) return notFound();
                if (!JOB_STATES.has(job.status)) return unavailable();
                return json({status: job.status});
            }
            return notFound();
        } catch {
            // Preserve privacy if the lookup, verifier, or serialization fails.
            return unavailable();
        }
    };
}

module.exports = {createPrivateBillingRead};
