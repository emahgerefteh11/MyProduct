'use strict';

// Browser integration rehearsal only. The fixture's clock and controls are
// not customer API routes. It binds exclusively to loopback.
const {mkdtempSync, readFileSync, rmSync} = require('node:fs');
const {createServer} = require('node:http');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {openTestHost, tenant, source} = require('./cur_test_host.cjs');

const directory = mkdtempSync(join(tmpdir(), 'atlas-browser-actions-'));
const period = '2026-01';
const reportTarget = {kind: 'report', source, period};
const model = {hour: 1, cloudCalls: 0};
let host, origin, queued, work, release, loseAck = true;
let submits = 0, disconnects = 0;
const error = status => Response.json({error: 'Fixture action unavailable.'}, {status});
const page = `<!doctype html><html lang="en"><meta charset="utf-8">
<script src="/billing_refresh_view.js"></script>
<button id="load">Load current selection</button>
<button id="refresh" disabled>Refresh</button>
<button id="check" disabled>Check refresh</button>
<button id="disconnect" disabled>Disconnect</button>
<output id="status">hidden</output><output id="accepted"></output>
<script>
const at = id => document.getElementById(id);
const action = (kind, fields, signal, headers = {}) => fetch('/' + kind + '?' + new URLSearchParams(fields), {
    method: 'POST', signal, headers: {'X-CSRF-Token': 'fixture-a', ...headers},
}).then(decodeBillingActionResponse);
async function read(path, fields, signal) {
    const response = await fetch(path + '?' + new URLSearchParams(fields), {signal});
    if ([401, 403, 404].includes(response.status)) return {status: 'denied'};
    if (!response.ok) throw new Error('Fixture read unavailable.');
    return response.json();
}
const view = createBillingRefreshView({
    submit: (scope, {requestId, signal}) => action('refresh', scope, signal, {'Idempotency-Key': requestId}),
    disconnect: (scope, {signal}) => action('disconnect', scope, signal),
    readJob: (jobId, {signal}) => read('/job', {jobId}, signal),
    readReport: (scope, {signal}) => read('/report', scope, signal),
    readSelection: (scope, {signal}) => read('/selection', scope, signal),
    render({status, accepted}) {
        at('status').textContent = status;
        at('accepted').textContent = accepted ? accepted.report.TotalBilledCost + ' | ' + accepted.lastSuccessfulSync : '';
        at('refresh').disabled = !['idle', 'retry-submit', 'completed', 'failed', 'canceled', 'selection-reloaded'].includes(status);
        at('check').disabled = !['queued', 'running', 'retry-status'].includes(status);
        at('disconnect').disabled = !['idle', 'retry-submit', 'completed', 'failed', 'canceled', 'selection-reloaded', 'queued', 'running'].includes(status);
    },
});
let selection = 0;
function clear() { selection++; view.clear(); }
at('load').onclick = async () => {
    clear();
    const turn = selection;
    const current = await read('/selection', {});
    if (turn === selection && current.scope) view.select(current.scope, current.accepted, current);
};
at('refresh').onclick = view.refresh;
at('check').onclick = view.check;
at('disconnect').onclick = view.stop;
addEventListener('pagehide', clear);
addEventListener('pageshow', event => { if (event.persisted) clear(); });
</script></html>`;

function requestFor(incoming, url) {
    const headers = new Headers(incoming.headers);
    const cookie = /(?:^|;\s*)atlas-action-session=([a-z0-9-]+)(?:;|$)/.exec(headers.get('Cookie') ?? '')?.[1];
    headers.set('Authorization', ({
        'fx-operator': 'Bearer fixture-a', 'fx-foreign': 'Bearer fixture-b',
    })[cookie] ?? 'Bearer revoked');
    return new Request(url, {method: incoming.method, headers});
}

async function control(command) {
    if (command === 'state') return Response.json({queued, submits, disconnects});
    if (command === 'viewer' || command === 'manager') host.setManage(command === 'manager');
    else if (command === 'advance-generation') host.advanceGeneration();
    else if (command === 'start' && queued && !work) {
        const jobId = queued;
        model.hour++;
        model.running = jobId;
        let reached;
        const fetching = new Promise(resolve => {reached = resolve;});
        model.hook = () => new Promise(resolve => {release = resolve; reached();});
        work = host.run({tenant, jobId}).finally(() => {
            model.running = null;
            model.hook = null;
        });
        await Promise.race([fetching, work]);
    } else if (command === 'finish' && work && release) {
        release();
        const outcome = await work;
        work = release = null;
        return Response.json(outcome);
    } else return error(409);
    return Response.json({ok: true});
}

const server = createServer(async (incoming, outgoing) => {
    try {
        const url = new URL(incoming.url, origin);
        const request = requestFor(incoming, url);
        let response;
        if (url.pathname === '/') response = new Response(page, {headers: {'Content-Type': 'text/html', 'Cache-Control': 'no-store'}});
        else if (url.pathname === '/billing_refresh_view.js') {
            response = new Response(readFileSync(join(__dirname, '../../billing_refresh_view.js')), {headers: {'Content-Type': 'text/javascript'}});
        } else if (url.pathname === '/report') response = await host.read(request, reportTarget);
        else if (url.pathname === '/job') response = await host.read(request, {kind: 'job', jobId: url.searchParams.get('jobId')});
        else if (url.pathname === '/selection') {
            response = await host.read(request, {kind: 'selection',
                source: url.searchParams.get('source') ?? source, period: url.searchParams.get('period') ?? period});
        } else if (url.pathname === '/refresh') {
            submits++;
            response = await host.write(request, {kind: 'refresh', source, period});
            if (response.status === 202) {
                queued = (await response.clone().json()).jobId;
                // The transaction accepted the first request, but the client
                // cannot tell that from this deliberate transient response.
                if (loseAck) {loseAck = false; response = error(503);}
            }
        } else if (url.pathname === '/disconnect') {
            disconnects++;
            response = await host.write(request, {kind: 'disconnect', source, generation: Number(url.searchParams.get('generation'))});
        } else if (url.pathname === '/__control' && incoming.method === 'POST' && incoming.headers['x-atlas-fixture-control'] === 'yes') {
            response = await control(url.searchParams.get('command'));
        } else response = error(404);
        outgoing.writeHead(response.status, Object.fromEntries(response.headers));
        outgoing.end(await response.text());
    } catch (error) {
        process.stderr.write(String(error) + '\n');
        if (!outgoing.headersSent) outgoing.writeHead(500);
        outgoing.end();
    }
});
process.on('exit', () => {host?.close(); rmSync(directory, {recursive: true, force: true});});
process.on('SIGTERM', () => process.exit(0));
server.listen(0, '127.0.0.1', async () => {
    origin = 'http://127.0.0.1:' + server.address().port;
    host = openTestHost(join(directory, 'billing.sqlite'), model, {hostOrigin: origin});
    host.seed();
    await host.run({tenant, jobId: 'first'});
    process.stdout.write(server.address().port + '\n');
});
