'use strict';

const {LIMITS} = require('./billing_import.js');
const {projectGcpStandardRows} = require('./local_gcp_projection.cjs');

const MAX_PAGES = 1000;
const jobKeys = ['projectId', 'jobId', 'location'];
const fits = n => typeof n === 'string' && /^(?:0|[1-9]\d{0,5})$/.test(n)
    && Number(n) >= 1 && Number(n) <= LIMITS.rows;
const sameJob = (a, b) => jobKeys.every(key => a?.[key] === b[key]);

// Offline reader contract. The caller supplies a verified completed job and
// an injected transport which decodes the BigQuery rows but preserves receipt
// metadata. No network, credential, IAM or query execution lives here.
async function collectGcpQueryProjection({job, readPage, billingAccount, invoiceMonth} = {}) {
    const ref = job?.jobReference;
    if (!jobKeys.every(key => typeof ref?.[key] === 'string' && ref[key].trim()) ||
        job.status?.state !== 'DONE' || job.status.errorResult || typeof readPage !== 'function') {
        throw new Error('Expected a successful completed GCP query and an explicit job reference.');
    }
    const expected = Object.freeze(Object.fromEntries(jobKeys.map(key => [key, ref[key]])));
    let token, total;
    const seen = new Set(), rows = [];
    for (let pageNo = 0; pageNo < MAX_PAGES; pageNo++) {
        const page = await readPage({jobReference: {...expected}, pageToken: token});
        if (!sameJob(page?.jobReference, expected) || page.jobComplete !== true) {
            throw new Error('GCP result page does not belong to the completed query.');
        }
        if (!pageNo || page.totalRows != null) {
            if (!fits(page.totalRows) || (pageNo && Number(page.totalRows) !== total)) {
                throw new Error('Expected one bounded GCP result row count that stays constant across pages.');
            }
            total = Number(page.totalRows);
        }
        if (!Array.isArray(page.rows) || !page.rows.length || rows.length + page.rows.length > total) {
            throw new Error('GCP result pages do not match the stated row count.');
        }
        for (const row of page.rows) rows.push(row);
        token = page.pageToken;
        if (token == null || token === '') {
            if (rows.length !== total) throw new Error('GCP query result ended before all rows were received.');
            return projectGcpStandardRows(rows, {billingAccount, invoiceMonth});
        }
        if (typeof token !== 'string' || token.length > 16384 || seen.has(token) || rows.length === total) {
            throw new Error('Invalid, repeated, or unexpected GCP query continuation.');
        }
        seen.add(token);
    }
    throw new Error('GCP query result exceeded the page limit.');
}
module.exports = {collectGcpQueryProjection};
