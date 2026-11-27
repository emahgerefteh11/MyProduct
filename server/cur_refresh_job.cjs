'use strict';

const {createLocalCurConnector} = require('../local_cur_connector.cjs');

const MONTH = /^(\d{4})-(0[1-9]|1[0-2])$/;
function monthlyScope(period) {
    const match = typeof period === 'string' && MONTH.exec(period);
    if (!match || Number(match[1]) < 1000 || Number(match[1]) > 9998) throw new Error('Expected a monthly CUR period.');
    const year = Number(match[1]), month = Number(match[2]);
    return [new Date(Date.UTC(year, month - 1, 1)).toISOString(), new Date(Date.UTC(year, month, 1)).toISOString()];
}

// Queue IDs are trusted, internal references; nothing here registers a public
// route. The store must atomically fence generation, mapping and revision and
// persist the outcome with the accepted snapshot. Free-standing process-local
// connector state is deliberately never read by the public/report path.
function createCurRefreshJobRunner({loadJob, openSource, commitJob, now}) {
    return async function run({tenant, jobId, signal}) {
        signal?.throwIfAborted();
        const job = await loadJob({tenant, jobId});
        if (!job || job.tenant !== tenant || job.jobId !== jobId) throw new Error('CUR refresh job is unavailable.');
        if (job.outcome) return job.outcome;
        const {source, period, generation, mappingVersion} = job;
        const [start, end] = monthlyScope(period);
        if (!source || !Number.isSafeInteger(generation) || generation < 1 ||
            !Number.isSafeInteger(mappingVersion) || mappingVersion < 0) {
            throw new Error('CUR refresh job is missing its connection version.');
        }
        const binding = {tenant, source, period, start, end, generation, mappingVersion};
        signal?.throwIfAborted();
        // openSource loads the current scoped credentials/client on the host;
        // it must never take a bucket, manifest key or credentials from a client.
        const access = await openSource(binding, {signal});
        if (!access || access.generation !== generation || access.mappingVersion !== mappingVersion) {
            throw new Error('CUR connection or mapping is no longer current.');
        }

        const connector = createLocalCurConnector({
            readStreamObject: access.readStreamObject,
            readSourceMapping: access.readSourceMapping,
            now,
        });
        const snapshot = await connector.refreshLatestAsync({tenant, source, manifestKey: access.manifestKey, signal});
        const range = snapshot.report.analysis.billingPeriod;
        if (!range.complete || range.start !== start || range.end !== end ||
            (snapshot.mappingVersion ?? 0) !== mappingVersion) {
            throw new Error('The validated CUR does not match the scheduled period or mapping.');
        }
        signal?.throwIfAborted();
        // If confirmation is lost, subsequent runs discover outcome via loadJob.
        // Do not translate an unknown commit error into terminal job failure.
        return commitJob({tenant, jobId, source, period, generation, mappingVersion, snapshot});
    };
}

module.exports = {createCurRefreshJobRunner};
