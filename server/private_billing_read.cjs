'use strict';

const {createHmac, timingSafeEqual} = require('node:crypto');
const {projectPrivateBillingOverview, projectPrivateBillingDetail, validDetail} = require('./private_billing_overview.cjs');

// Server-only boundary. The host must supply a verified, current session and
// tenant-scoped storage, not a tenant claim taken directly from the request.
// This module installs no routes and contains no credentials or cloud client.
const ACCEPTED_TOKEN = /^[a-f0-9]{64}$/;
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const PERIOD = /^\d{4}-(?:0[1-9]|1[0-2])$/;
const JOB_STATES = new Set(['queued', 'running', 'completed', 'failed', 'canceled']);
const headers = {'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'};
const json = (body, status = 200, extra = {}) => Response.json(body, {status, headers: {...headers, ...extra}});
const notFound = () => json({error: 'Not found.'}, 404);
const stale = () => json({error: 'Billing report changed. Reload the overview.'}, 409);
const unavailable = () => json({error: 'Private billing is unavailable. Try again.'}, 503);
const valid = (pattern, value) => typeof value === 'string' && pattern.test(value);

function reportEnvelope(snapshot) {
    const {report, acceptedAt} = snapshot;
    if (!report || report.sampleData !== false || typeof acceptedAt !== 'string' ||
        !Number.isFinite(Date.parse(acceptedAt))) throw new Error('Invalid accepted report.');
    return {report, lastSuccessfulSync: acceptedAt};
}

function revisionToken(s, key) {
    const {tenant, source, period, assembly, revision, mappingVersion} = s;
    if (!valid(ID, assembly) || !valid(ID, revision) || !Number.isSafeInteger(mappingVersion) || mappingVersion < 0) {
        throw new Error('Invalid accepted revision.');
    }
    return createHmac('sha256', key)
        .update(JSON.stringify(['private-billing-v1', tenant, source, period, assembly, revision, mappingVersion]))
        .digest('hex');
}

// Every read callback must authorize the current actor and read the requested
// record in one scoped operation. Session identity alone is not a billing grant.
// Missing records and absent read grants both return null. A disconnected source
// may retain a report for authorized readers but must never enable writes.
function matchesScope(row, {tenant, source, period}) {
    return row.tenant === tenant && row.source === source && row.period === period;
}

function createPrivateBillingRead({resolveSession, readReport, readJob, readSelection, acceptedTokenKey,
    readDetailMetadata, readDetailBody}) {
    if (acceptedTokenKey != null && (!Buffer.isBuffer(acceptedTokenKey) || acceptedTokenKey.length < 32)) {
        throw new Error('Invalid accepted revision key.');
    }
    const key = acceptedTokenKey == null ? null : Buffer.from(acceptedTokenKey);
    return async function handle(request, target) {
        if (request.method !== 'GET') return json({error: 'Method not allowed.'}, 405, {Allow: 'GET'});

        try {
            const session = await resolveSession(request);
            const tenant = session?.tenantId, actor = session?.actor;
            if (!valid(ID, tenant) || !valid(ID, actor)) return json({error: 'Sign-in required.'}, 401);

            if (target?.kind === 'report' && valid(ID, target.source) && valid(PERIOD, target.period)) {
                const snapshot = await readReport({tenant, actor, source: target.source, period: target.period});
                if (snapshot == null) return notFound();
                return json(reportEnvelope(snapshot));
            }
            const kind = target?.kind;
            if (kind === 'detail' && key && typeof readDetailMetadata === 'function' && typeof readDetailBody === 'function' &&
                valid(ID, target.source) && valid(PERIOD, target.period)) {
                const scope = {tenant, actor, source: target.source, period: target.period};
                const snapshot = await readDetailMetadata(scope);
                if (snapshot == null) return notFound();
                if (!matchesScope(snapshot, scope)) return unavailable();
                if (!validDetail(target.dimension, target.offset) || !valid(ACCEPTED_TOKEN, target.acceptedRevision)) {
                    return json({error: 'An accepted revision and detail selection are required.'}, 400);
                }
                const acceptedRevision = revisionToken(snapshot, key);
                if (!timingSafeEqual(Buffer.from(target.acceptedRevision), Buffer.from(acceptedRevision))) return stale();

                // The body lookup must recheck the live grant and exact tuple
                // in the same operation that reads report bytes.
                const {assembly, revision, mappingVersion} = snapshot;
                const matched = await readDetailBody({...scope, assembly, revision, mappingVersion});
                if (matched == null) {
                    const current = await readDetailMetadata(scope);
                    if (current == null) return notFound();
                    if (!matchesScope(current, scope) || revisionToken(current, key) === acceptedRevision) return unavailable();
                    return stale();
                }
                if (!matchesScope(matched, scope) || revisionToken(matched, key) !== acceptedRevision) return unavailable();
                const {report} = reportEnvelope(matched);
                const detail = projectPrivateBillingDetail(report, target.dimension, target.offset);
                return detail == null ? notFound() : json({acceptedRevision, detail});
            }
            if ((kind === 'selection' || (kind === 'overview' && key)) && valid(ID, target.source) && valid(PERIOD, target.period)) {
                const {source, period} = target;
                const scope = {tenant, actor, source, period};
                const selection = await readSelection(scope);
                if (selection == null) return notFound();
                const {generation, active, canManageBilling, snapshot} = selection;
                if (!matchesScope(selection, scope) || !Number.isSafeInteger(generation) || generation < 1 ||
                    typeof active !== 'boolean' || typeof canManageBilling !== 'boolean' ||
                    (snapshot != null && !matchesScope(snapshot, scope))) return unavailable();
                if (kind === 'overview') {
                    if (snapshot == null) return notFound();
                    return json({acceptedRevision: revisionToken(snapshot, key), overview: projectPrivateBillingOverview(snapshot)});
                }
                return json({scope: {source, period, generation},
                    connectionStatus: active ? 'active' : 'disconnected',
                    canManageBilling: active && canManageBilling,
                    accepted: snapshot == null ? null : reportEnvelope(snapshot)});
            }
            if (target?.kind === 'job' && valid(ID, target.jobId)) {
                const job = await readJob({tenant, actor, jobId: target.jobId});
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
