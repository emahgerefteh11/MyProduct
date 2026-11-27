'use strict';

const assert = require('node:assert/strict');
const {existsSync, writeSync} = require('node:fs');
const {performance} = require('node:perf_hooks');
const {openTestHost, tenant} = require('./cur_test_host.cjs');

async function main() {
    const [, , file, gate, size] = process.argv;
    assert.ok(file && gate && ['small', 'large'].includes(size));
    const started = performance.now();
    let transactionReady, released;
    const host = openTestHost(file, {
        hour: 2, cloudCalls: 0, largeFixture: size === 'large',
        afterJobResultBeforeCommit() {
            transactionReady = performance.now();
            writeSync(1, 'holding\n');
            const wait = new Int32Array(new SharedArrayBuffer(4));
            while (!existsSync(gate)) {
                if (performance.now() - transactionReady > 15_000) throw new Error('Test write barrier timed out.');
                Atomics.wait(wait, 0, 0, 10);
            }
            released = performance.now();
        },
    });
    try {
        const result = await host.run({tenant, jobId: 'correction'});
        writeSync(1, JSON.stringify({status: result.status, timeToBarrierMs: transactionReady - started,
            commitAfterReleaseMs: performance.now() - released, peakRssKiB: process.resourceUsage().maxRSS}) + '\n');
    } finally {
        host.close();
    }
}

main().catch(e => { console.error(e); process.exitCode = 1; });
