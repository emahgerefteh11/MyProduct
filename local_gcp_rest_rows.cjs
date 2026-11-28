'use strict';

const {LIMITS} = require('./billing_import.js');

// Exact shape of the candidate one-invoice query. Reject schema drift rather
// than using a cost or credit field from the wrong column.
const FIELDS = [
    ['billing_account_id'], ['invoice', [['month']]], ['currency'],
    ['service', [['description']]], ['cost'], ['credits', [['amount']], true],
    ['cost_type'], ['usage_start_time'], ['usage_end_time'],
];

function validateSchema(actual, expected) {
    if (!Array.isArray(actual) || actual.length !== expected.length) throw new Error('Unexpected GCP query result schema.');
    expected.forEach(([name, children, repeated], i) => {
        const field = actual[i], mode = field?.mode ?? 'NULLABLE';
        if (field?.name !== name || (repeated ? mode !== 'REPEATED' : !['NULLABLE', 'REQUIRED'].includes(mode)) ||
            !(children ? ['RECORD', 'STRUCT'].includes(field.type) : field.type === 'STRING')) {
            throw new Error('Unexpected GCP query result schema.');
        }
        if (children) validateSchema(field.fields, children);
    });
}

function unwrap(cell) {
    if (cell == null || typeof cell !== 'object' || !Object.hasOwn(cell, 'v')) {
        throw new Error('Malformed GCP REST result row.');
    }
    return cell.v;
}

function decodeRecord(record, fields) {
    if (!Array.isArray(record?.f) || record.f.length !== fields.length) {
        throw new Error('Malformed GCP REST result row.');
    }
    return Object.fromEntries(fields.map(([name, children, repeated], index) => {
        const value = unwrap(record.f[index]);
        if (repeated) {
            if (!Array.isArray(value) || value.length > 128) throw new Error('Malformed GCP REST repeated field.');
            return [name, Array.from(value, cell => decodeRecord(unwrap(cell), children))];
        }
        if (value === null) return [name, null];
        if (children) return [name, decodeRecord(value, children)];
        if (typeof value !== 'string' || value.length > 4096) throw new Error('Malformed GCP REST scalar field.');
        return [name, value];
    }));
}

// Bound to one sequential collection by its caller. The transport remains
// responsible for authenticated job retrieval and raw HTTP byte limits.
function makeGcpRestPageReader(readRawPage) {
    if (typeof readRawPage !== 'function') throw new Error('A GCP raw page reader is required.');
    let validated = false;
    return async request => {
        const page = await readRawPage(request);
        if (page?.jobComplete !== true) return page;
        if (!validated || page.schema != null) {
            validateSchema(page.schema?.fields, FIELDS);
            validated = true;
        }
        if (!Array.isArray(page.rows) || page.rows.length > LIMITS.rows) throw new Error('Malformed GCP REST result page.');
        return {jobReference: page.jobReference, jobComplete: page.jobComplete, totalRows: page.totalRows,
            pageToken: page.pageToken, rows: Array.from(page.rows, record => decodeRecord(record, FIELDS))};
    };
}

module.exports = {makeGcpRestPageReader};
