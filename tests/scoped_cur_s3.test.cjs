'use strict';

const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {Readable} = require('node:stream');
const test = require('node:test');
const {bindKeptCurS3Reader} = require('../server/scoped_cur_s3.cjs');
const {createLocalCurConnector} = require('../local_cur_connector.cjs');

const directory = 'fixture/cur/20260101-20260201';
const manifestKey = directory + '/atlas-Manifest.json';
const assemblyId = '20260202T010000Z';
const part = directory + '/' + assemblyId + '/atlas-00001.csv.gz';
const etag = '"sample"';
const scope = {tenant: 'fixture-tenant', source: 'fixture-source'};
const bucket = 'fixture-cur-bucket';
const expectedBucketOwner = '111122223333';

test('authorization scope is checked before the SDK call and only permits configured manifest and CUR parts', async () => {
    const calls = [];
    const read = bindKeptCurS3Reader({
        ...scope, bucket, expectedBucketOwner, manifestKey,
        sendGetObject: async (...args) => {calls.push(args);},
    });
    const signal = new AbortController().signal;
    await read({...scope, key: manifestKey, ifMatch: etag, signal, bucket: 'ignored'});
    await read({...scope, key: part, signal});
    assert.deepEqual(calls, [
        [{Bucket: bucket, Key: manifestKey, ExpectedBucketOwner: expectedBucketOwner, IfMatch: etag}, {abortSignal: signal}],
        [{Bucket: bucket, Key: part, ExpectedBucketOwner: expectedBucketOwner}, {abortSignal: signal}],
    ]);
    for (const changed of [
        {tenant: 'other'}, {source: 'other'}, {key: manifestKey.replace('20260101', '20260201')},
        {key: directory + '/atlas.csv.gz'}, {key: directory + '/' + assemblyId + '/other-00001.csv.gz'},
        {key: part + '/extra'}, {key: part.replace('.csv.gz', '.parquet')},
        {key: part, ifMatch: etag}, {ifMatch: '*'},
    ]) {
        await assert.rejects(read({...scope, key: manifestKey, ...changed}), /outside the configured source scope/);
    }
    assert.equal(calls.length, 2);
});

test('synthetic CUR refresh uses the scoped bridge and retains the last report when read permission fails', async () => {
    const calls = [];
    const response = {
        assemblyId, reportName: 'atlas', reportKeys: [part],
        billingPeriod: {start: '20260101T000000.000Z', end: '20260201T000000.000Z'},
    };
    let deny = false;
    const readStreamObject = bindKeptCurS3Reader({
        ...scope, bucket, expectedBucketOwner, manifestKey,
        sendGetObject: async input => {
            calls.push(input);
            if (deny) throw Object.assign(new Error('manifest unavailable'), {$metadata: {httpStatusCode: 403}});
            if (input.Key === manifestKey) return {Body: Readable.from([Buffer.from(JSON.stringify(response))]), ETag: etag};
            const raw = readFileSync(join(__dirname, 'fixtures', 'cur', 'synthetic.csv.gz'));
            return {Body: Readable.from([raw])};
        },
    });
    const connector = createLocalCurConnector({readStreamObject});
    const accepted = await connector.refreshLatestAsync({...scope, manifestKey});
    assert.equal(accepted.report.TotalBilledCost, 100);
    assert.deepEqual(calls.map(({Key, IfMatch}) => [Key, IfMatch]), [[manifestKey, undefined], [part, undefined], [manifestKey, etag]]);
    deny = true;
    await assert.rejects(connector.refreshLatestAsync({...scope, manifestKey}), /manifest unavailable/);
    assert.deepEqual(connector.get({...scope, start: '2026-01-01T00:00:00.000Z', end: '2026-02-01T00:00:00.000Z'}), accepted);
});
