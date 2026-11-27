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

function createPrivateBillingRead({resolveSession, readReport, readJob}) {
    return async function handle(request, target) {
        if (request.method !== 'GET') return json({error: 'Method not allowed.'}, 405, {Allow: 'GET'});

        try {
            const tenant = (await resolveSession(request))?.tenantId;
            if (!valid(ID, tenant)) return json({error: 'Sign-in required.'}, 401);

            if (target?.kind === 'report' && valid(ID, target.source) && valid(PERIOD, target.period)) {
                const snapshot = await readReport({tenant, source: target.source, period: target.period});
                if (snapshot == null) return notFound();
                const {report, acceptedAt} = snapshot;
                if (!report || report.sampleData !== false || !Number.isFinite(Date.parse(acceptedAt))) return unavailable();
                return json({report, lastSuccessfulSync: acceptedAt});
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
