'use strict';

// Child process for disposable synthetic SQLite files only. It terminates
// itself at one instructed boundary, never a separately selected process.
const assert = require('node:assert/strict');
const {writeSync} = require('node:fs');
const {openTestHost, tenant} = require('./cur_test_host.cjs');

async function main() {
    const [, , file, point] = process.argv;
    assert.ok(file && ['before-commit', 'after-commit'].includes(point));
    const exitAbruptly = () => {
        writeSync(1, point + '\n');
        process.kill(process.pid, 'SIGKILL');
    };
    const h = {hour: 2, cloudCalls: 0,
        [point === 'before-commit' ? 'afterJobResultBeforeCommit' : 'afterCommit']: exitAbruptly};
    const host = openTestHost(file, h);
    await host.run({tenant, jobId: 'correction'});
    throw new Error('The intended crash point was not reached.');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
