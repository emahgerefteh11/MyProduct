'use strict';

// Test-only loopback server. Never mount this as a customer or demo endpoint.
const {readFileSync} = require('node:fs');
const {createServer} = require('node:http');
const {join} = require('node:path');
const {gzipSync} = require('node:zlib');
const {createLocalCurConnector} = require('../../local_cur_connector.cjs');
const {createPrivateBillingRead} = require('../../server/private_billing_read.cjs');

const root = join(__dirname, '../..');
const manifest = require('./cur/manifest.json');
const csv = readFileSync(join(__dirname, 'cur/synthetic.csv'), 'utf8');
const scope = {tenant: 'tenant-a', source: 'aws'};
const period = '2026-01';
const connector = createLocalCurConnector({
    now: () => new Date('2026-02-02T02:00:00Z'),
    readObject({key}) {
        if (key !== manifest.reportKeys[0]) throw new Error('Unexpected fixture object.');
        return gzipSync(csv.replace(',30,USD,', ',35,USD,'));
    },
});
const accepted = connector.refresh({...scope, manifest});
const sessions = new Map([
    ['fx-965ed1', {tenantId: scope.tenant}],
    ['fx-ab64d3', {tenantId: 'tenant-b'}],
    ['fx-1186c0', {tenantId: scope.tenant, revoked: true}],
]);
const readPrivate = createPrivateBillingRead({
    resolveSession(request) {
        const token = /(?:^|;\s*)atlas-fixture-session=([a-z0-9-]+)(?:;|$)/.exec(request.headers.get('Cookie') ?? '')?.[1];
        const session = sessions.get(token);
        return session?.revoked ? null : session;
    },
    readReport(request) {
        return request.tenant === scope.tenant && request.source === scope.source &&
            request.period === period ? accepted : null;
    },
    readJob: () => null,
});

const fixturePath = '/__fixture/private-browser';
const reportPath = fixturePath + '/report';
const fixtureHtml = `<!doctype html><html lang="en"><meta charset="utf-8">
<script src="/report_store.js"></script><script src="/report_view_session.js"></script>
<button id="connected">Open connected report</button><button id="upload">Open local report</button>
<output id="fixture-output">hidden</output>
<script>
const view = createReportViewSession({
    async readConnected(_, signal) {
        const response = await fetch('${reportPath}?source=aws&period=${period}&tenant=tenant-a', {signal});
        if (!response.ok) return null;
        return (await response.json()).report;
    },
    readUpload: () => ReportStore.get(),
    render: ({status, source, report}) => {
        document.querySelector('#fixture-output').textContent = report ? source + ':' + report.TotalBilledCost : status;
    },
});
document.querySelector('#connected').onclick = () => view.openConnected();
document.querySelector('#upload').onclick = () => view.openUpload();
addEventListener('pagehide', view.clear);
addEventListener('pageshow', event => { if (event.persisted) view.clear(); });
</script></html>`;

const publicFiles = new Set([
    '/import.html', '/dashboard.html', '/atlas.css', '/dashboard.css', '/import.css', '/motion.css',
    '/import.js', '/billing_worker.js', '/billing_import.js', '/report_store.js',
    '/report_view_session.js', '/dashboard_ui.js',
]);
const types = {html: 'text/html', css: 'text/css', js: 'text/javascript'};
const server = createServer(async (incoming, outgoing) => {
    try {
        const url = new URL(incoming.url, 'http://127.0.0.1');
        const path = url.pathname;
        if (path === reportPath) {
            const response = await readPrivate(new Request(url, {method: incoming.method, headers: incoming.headers}), {
                kind: 'report', source: url.searchParams.get('source'), period: url.searchParams.get('period'),
            });
            outgoing.writeHead(response.status, Object.fromEntries(response.headers));
            outgoing.end(await response.text());
        } else if (path === fixturePath) {
            outgoing.writeHead(200, {'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store'});
            outgoing.end(fixtureHtml);
        } else if (publicFiles.has(path)) {
            const type = types[path.split('.').at(-1)];
            outgoing.writeHead(200, {'Content-Type': type + '; charset=utf-8'});
            outgoing.end(readFileSync(join(root, path.slice(1))));
        } else {
            outgoing.writeHead(404).end();
        }
    } catch {
        if (!outgoing.headersSent) outgoing.writeHead(500);
        outgoing.end();
    }
});
server.listen(0, '127.0.0.1', () => process.stdout.write(server.address().port + '\n'));
