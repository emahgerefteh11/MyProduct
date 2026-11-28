"""Browser acceptance tests. Run separately after installing Playwright Chromium."""
import csv
import faulthandler
import functools
import io
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import unittest
import zipfile

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class BrowserImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f'http://127.0.0.1:{cls.server.server_port}'
        cls.playwright = sync_playwright().start()
        options = {'headless': True}
        if os.environ.get('ATLAS_CHROMIUM_PATH'):
            options['executable_path'] = os.environ['ATLAS_CHROMIUM_PATH']
        cls.browser = cls.playwright.chromium.launch(**options)
        cls.artifacts = ROOT / 'test-artifacts'
        cls.artifacts.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US')
        # No third-party scripts or data requests are needed to import or view costs.
        self.context.route('**/*', lambda route: route.continue_() if urlparse(route.request.url).netloc == urlparse(self.origin).netloc else route.abort())
        self.page = self.context.new_page()
        self.js_errors = []
        self.page.on('pageerror', lambda error: self.js_errors.append(str(error)))
        self.page.goto(self.origin + '/import.html')

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.js_errors, [], 'Browser JavaScript errors')

    def choose(self, files):
        self.page.locator('#billing-files').set_input_files(files)
        self.page.locator('#read-files').click()
        expect(self.page.locator('#step-2')).to_be_visible()

    def validate(self):
        self.page.locator('#validate').click()
        expect(self.page.locator('#step-3')).to_be_visible()
        expect(self.page.locator('#status')).to_be_hidden()

    def accept(self):
        expect(self.page.locator('#open-report')).to_be_disabled()
        self.page.locator('#acknowledge').check()
        self.page.locator('#open-report').click()
        self.page.wait_for_url('**/dashboard.html?source=import')
        expect(self.page.locator('#mode-badge-header')).to_contain_text('REPORT DATA')

    def open_adoption_fixture(self):
        aws = 'lineItem/UnblendedCost,lineItem/ResourceId,lineItem/UsageAccountId,product/regionCode,lineItem/ProductCode,lineItem/CurrencyCode\n10,i-0123456789abcdef0,123456789012,us-east-1,AmazonEC2,USD\n4,vol-0123456789abcdef0,123456789012,us-east-1,AmazonEC2,USD\n2,web-prod,123456789012,us-east-1,AmazonEC2,USD\n-1,,123456789012,us-east-1,AmazonEC2,USD\n'
        gcp = 'cost,service.description,resource.global_name,currency\n3,Compute Engine,//compute.googleapis.com/projects/atlas-prod/zones/us-central1-a/instances/web-prod,USD\n'
        self.choose([{'name': 'aws.csv', 'mimeType': 'text/csv', 'buffer': aws.encode()}, {'name': 'gcp.csv', 'mimeType': 'text/csv', 'buffer': gcp.encode()}])
        self.validate()
        self.accept()
        self.page.locator('.page-actions').get_by_role('link', name='Adopt into IaC').click()
        expect(self.page.locator('#workspace')).to_be_visible()
        expect(self.page.locator('#metrics')).to_contain_text('4 resources')

    def test_adoption_package_download_and_discovery_progress(self):
        self.open_adoption_fixture()
        expect(self.page.locator('#coverage-note')).to_contain_text('1 billing row(s)')
        self.page.locator('#select-ready').click()
        expect(self.page.locator('#selection-summary')).to_contain_text('3 selected')
        with self.page.expect_download() as info:
            self.page.locator('#download-package').click()
        with zipfile.ZipFile(info.value.path()) as package:
            self.assertIsNone(package.testzip())
            self.assertTrue(all(entry.date_time[:3] == (1980, 1, 1) for entry in package.infolist()))
            self.assertEqual(set(package.namelist()), {'candidates.json', 'adoption_tool.py', 'README.md'})
            self.assertEqual(package.read('adoption_tool.py'), (ROOT / 'adoption_tool.py').read_bytes())
            manifest = json.loads(package.read('candidates.json'))
        self.assertEqual(len(manifest['resources']), 3)
        self.assertEqual(manifest['resources'][0]['account'], '123456789012')
        self.page.screenshot(path=str(self.artifacts / 'adoption-desktop.png'), full_page=True)
        inventory = {'schemaVersion': 1, 'kind': 'atlas-adoption-inventory', 'resources': [{**manifest['resources'][0], 'status': 'imported', 'message': 'Clean plan <img src=x onerror=alert(1)>', 'checkedAt': '2026-10-23T00:00:00Z'}]}
        self.page.locator('#inventory-file').set_input_files({'name': 'inventory.json', 'mimeType': 'application/json', 'buffer': json.dumps(inventory).encode()})
        expect(self.page.locator('#inventory-summary')).to_contain_text('1 imported')
        expect(self.page.locator('#resource-rows')).to_contain_text('Clean plan <img')
        self.assertEqual(self.page.locator('#resource-rows img').count(), 0)
        expect(self.page.locator('#metrics')).to_contain_text('1 imported')
        self.page.locator('#select-ready').click()
        expect(self.page.locator('#selection-summary')).to_contain_text('2 selected')
        expect(self.page.locator('#resource-rows input').first).to_be_disabled()

    def test_adoption_scope_fill_invalid_results_and_mobile(self):
        report = {'sampleData': False, 'analysis': {'currency': 'USD'}, 'resourceInventory': [{'provider': 'aws', 'resourceId': 'i-0123456789abcdef0', 'cost': 10, 'sourceHashes': []}]}
        self.page.goto(self.origin + '/adoption.html')
        expect(self.page.locator('#status')).to_be_hidden()
        self.page.locator('#report-file').set_input_files({'name': 'report.json', 'mimeType': 'application/json', 'buffer': json.dumps(report).encode()})
        expect(self.page.locator('#metrics')).to_contain_text('0 ready')
        self.page.locator('#scope-options summary').click()
        self.page.locator('#aws-account').fill('123456789012')
        self.page.locator('#aws-region').fill('us-east-1')
        self.page.locator('#apply-scope').click()
        expect(self.page.locator('#metrics')).to_contain_text('1 ready')
        self.page.locator('#select-ready').click()
        expect(self.page.locator('#download-package')).to_be_enabled()
        self.page.locator('#inventory-file').set_input_files({'name': 'state.json', 'mimeType': 'application/json', 'buffer': b'{"version":4,"resources":[]}'})
        expect(self.page.locator('#error')).to_contain_text('Raw Terraform plan/state')
        expect(self.page.locator('#download-package')).to_be_enabled()
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
        self.page.screenshot(path=str(self.artifacts / 'adoption-mobile.png'), full_page=True)
        self.page.locator('#report-file').set_input_files({'name': 'old.json', 'mimeType': 'application/json', 'buffer': b'{"resourceNames":[]}'})
        expect(self.page.locator('#error')).to_contain_text('lacks scoped resource inventory')
        expect(self.page.locator('#metrics')).to_contain_text('1 ready')

    def test_adoption_sample_report_cannot_download_real_import_package(self):
        report = {'sampleData': True, 'resourceInventory': [{'provider': 'aws', 'resourceId': 'i-0123456789abcdef0', 'account': '123456789012', 'region': 'us-east-1', 'cost': 10}]}
        self.page.goto(self.origin + '/adoption.html')
        expect(self.page.locator('#status')).to_be_hidden()
        self.page.locator('#report-file').set_input_files({'name': 'sample.json', 'mimeType': 'application/json', 'buffer': json.dumps(report).encode()})
        expect(self.page.locator('#report-summary')).to_contain_text('Sample report: preview only')
        self.page.locator('#select-ready').click()
        expect(self.page.locator('#download-package')).to_be_disabled()
        expect(self.page.locator('#resource-rows input')).to_be_disabled()

    def test_cur_parity_fixture_reaches_the_dashboard(self):
        self.choose([ROOT / 'tests' / 'fixtures' / 'cur' / 'synthetic.csv'])
        expect(self.page.locator('#format-0')).to_have_value('aws')
        self.validate()
        expect(self.page.locator('#preview-metrics')).to_contain_text('$100.00')
        self.accept()
        expect(self.page.locator('#kpi-total')).to_have_text('$100.00')
        self.page.locator('.report-actions summary').click()
        with self.page.expect_download() as download:
            self.page.locator('#download-imported-report').click()
        report = json.loads(Path(download.value.path()).read_text())
        self.assertEqual(report['analysis']['currency'], 'USD')
        self.assertEqual(report['analysis']['billingPeriod']['start'], '2026-01-01T00:00:00.000Z')
        self.assertEqual(report['allocation']['unallocatedProductCost'], 30)
        self.assertEqual(next(p for p in report['products'] if p['Product'] == 'Payments')['BilledCost'], 70)

    def test_preview_explains_rounded_cost_without_negative_zero(self):
        for currency, raw, delta in [
            ('KWD', '1.2344', '−0.0004 KWD'),
            ('JPY', '10.49', '−0.49 JPY'),
            ('USD', '1.2344', '−0.0044 USD'),
            ('USD', '1.2300', 'none'),
            ('USD', '1.2299999999', 'less than 0.00000001 USD (report is higher)'),
        ]:
            self.page.goto(self.origin + '/import.html')
            self.choose([{'name': 'focus.csv', 'mimeType': 'text/csv', 'buffer':
                          f'BilledCost,BillingCurrency,Product\n{raw},{currency},Payments\n'.encode()}])
            self.validate()
            evidence = self.page.locator('#reconciliation')
            expect(evidence).to_contain_text(f'Rounding difference: {delta}.')
            expect(evidence).to_contain_text('not completeness against your cloud invoice')

    def test_saved_report_keeps_rounding_evidence_after_reload(self):
        self.choose([{'name': 'focus.csv', 'mimeType': 'text/csv', 'buffer':
                      b'BilledCost,BillingCurrency,Product\n1.2344,KWD,Payments\n'}])
        self.validate()
        expect(self.page.locator('#reconciliation')).to_contain_text('Rounding difference: −0.0004 KWD.')
        self.accept()
        for _ in range(2):
            self.page.locator('.quality-panel summary').click()
            expect(self.page.locator('#report-quality')).to_contain_text('Rounding difference: −0.0004 KWD.')
            self.page.reload()
        self.page.evaluate('''async () => {
            const report = await ReportStore.get();
            delete report.analysis.reconciliation.exactDifference;
            await ReportStore.save(report);
        }''')
        self.page.reload()
        self.page.locator('.quality-panel summary').click()
        expect(self.page.locator('#report-quality')).to_contain_text('Rounding difference: −0.0004 KWD.')

    def test_currency_specific_precision_reaches_the_dashboard(self):
        for currency, raw, total, display in [
            ('KWD', '1.2344', 1.234, '1.234'),
            ('JPY', '10.49', 10, '10'),
        ]:
            self.choose([{'name': 'focus.csv', 'mimeType': 'text/csv', 'buffer':
                          f'BilledCost,BillingCurrency,Product\n{raw},{currency},Payments\n'.encode()}])
            self.validate()
            expect(self.page.locator('#preview-metrics')).to_contain_text(display)
            self.accept()
            expected = self.page.evaluate(
                '([currency, total]) => new Intl.NumberFormat("en-US", {style: "currency", currency}).format(total)',
                [currency, total])
            expect(self.page.locator('#kpi-total')).to_have_text(expected)
            menu = self.page.locator('.report-actions')
            if menu.get_attribute('open') is None:
                menu.locator('summary').click()
            with self.page.expect_download() as download:
                self.page.locator('#download-imported-report').click()
            report = json.loads(Path(download.value.path()).read_text())
            self.assertEqual(report['TotalBilledCost'], total)
            self.assertEqual(report['products'][0]['BilledCost'], total)
            self.page.goto(self.origin + '/import.html')

    def test_demo_correction_replaces_the_original_import(self):
        for file, total, unallocated in [
            (ROOT / 'tests' / 'fixtures' / 'cur' / 'synthetic.csv', 100, 30),
            (ROOT / 'docs' / 'demo' / 'aws-corrected.csv', 105, 35),
        ]:
            self.choose([file])
            expect(self.page.locator('#format-0')).to_have_value('aws')
            self.validate()
            expect(self.page.locator('#preview-metrics')).to_contain_text(f'${total}.00')
            self.accept()
            expect(self.page.locator('#kpi-total')).to_have_text(f'${total}.00')
            self.page.locator('.report-actions summary').click()
            with self.page.expect_download() as download:
                self.page.locator('#download-imported-report').click()
            report = json.loads(Path(download.value.path()).read_text())
            self.assertEqual(report['allocation']['unallocatedProductCost'], unallocated)
            self.assertEqual(next(p for p in report['products'] if p['Product'] == 'Payments')['BilledCost'], 70)
            if total == 100:
                menu = self.page.locator('.report-actions')
                if menu.get_attribute('open') is None:
                    menu.locator('summary').click()
                self.page.get_by_role('link', name='Import a new report').click()
                self.page.wait_for_url('**/import.html')
        self.page.reload()
        expect(self.page.locator('#kpi-total')).to_have_text('$105.00')

    def test_focus_nullable_and_conflicting_ownership_tags(self):
        rows = 'BilledCost,BillingCurrency,ProviderName,Tags\n'
        rows += '20,USD,AWS,"{""Product"":""Payments"",""Platform"":null}"\n'
        rows += '10,USD,AWS,"{""Product"":""First"",""product"":""Second""}"\n'
        self.choose([{'name': 'focus.csv', 'mimeType': 'text/csv', 'buffer': rows.encode()}])
        expect(self.page.locator('#format-0')).to_have_value('focus')
        self.validate()
        warnings = self.page.locator('#warning-list')
        expect(warnings).to_contain_text('ownership tags have conflicting values')
        expect(warnings).not_to_contain_text('tags could not be parsed')
        self.accept()
        expect(self.page.locator('#kpi-total')).to_have_text('$30.00')
        self.page.locator('.report-actions summary').click()
        with self.page.expect_download() as download:
            self.page.locator('#download-imported-report').click()
        report = json.loads(Path(download.value.path()).read_text())
        self.assertEqual(report['allocation']['unallocatedProductCost'], 10)
        self.assertEqual(next(p for p in report['products'] if p['Product'] == 'Payments')['BilledCost'], 20)

    def test_connected_view_contract_keeps_private_and_uploaded_reports_separate(self):
        self.choose([ROOT / 'tests' / 'fixtures' / 'cur' / 'synthetic.csv'])
        self.validate()
        self.accept()
        expect(self.page.locator('#kpi-total')).to_have_text('$100.00')

        # The future private route is simulated here. No account, endpoint or
        # login is exposed by the public dashboard; storage is real IndexedDB.
        path = self.origin + '/__atlas_fixture_connected.html'
        html = """<!doctype html><html lang="en"><meta charset="utf-8">
            <script src="/report_store.js"></script>
            <script src="/report_view_session.js"></script>
            <output id="state">hidden</output>
            <script>
            const pending = [];
            let uploadReads = 0;
            const view = createReportViewSession({
                readConnected: tenant => new Promise(resolve => pending.push({tenant, resolve})),
                readUpload: () => { uploadReads++; return ReportStore.get(); },
                render: ({status, source, report}) => {
                    document.querySelector('#state').textContent = report
                        ? source + ':' + report.TotalBilledCost : status;
                },
            });
            addEventListener('pagehide', view.clear);
            addEventListener('pageshow', event => { if (event.persisted) view.clear(); });
            </script></html>"""
        self.page.route(path, lambda route: route.fulfill(content_type='text/html', body=html))
        self.page.goto(path)
        state = self.page.locator('#state')
        expect(state).to_have_text('hidden')
        self.page.evaluate("() => { void view.openConnected('a'); pending[0].resolve({TotalBilledCost:105}); }")
        expect(state).to_have_text('connected:105')
        self.assertEqual(self.page.evaluate('uploadReads'), 0)

        self.page.evaluate("""() => {
            void view.openConnected('a');
            void view.openConnected('b');
            pending[2].resolve({TotalBilledCost:50});
            pending[1].resolve({TotalBilledCost:105});
        }""")
        expect(state).to_have_text('connected:50')
        self.page.evaluate("""() => {
            void view.openConnected('b');
            view.clear();
            pending[3].resolve({TotalBilledCost:50});
        }""")
        expect(state).to_have_text('hidden')

        self.page.evaluate("""() => {
            void view.openConnected('a');
            dispatchEvent(new PageTransitionEvent('pagehide', {persisted:true}));
            dispatchEvent(new PageTransitionEvent('pageshow', {persisted:true}));
            pending[4].resolve({TotalBilledCost:105});
        }""")
        expect(state).to_have_text('hidden')
        self.page.goto(self.origin + '/import.html')
        self.page.go_back()
        expect(state).to_have_text('hidden')
        self.page.evaluate('view.openUpload()')
        expect(state).to_have_text('upload:100')
        self.page.goto(self.origin + '/dashboard.html?source=import')
        expect(self.page.locator('#kpi-total')).to_have_text('$100.00')

    def start_node_fixture(self, fixture):
        proc = subprocess.Popen(
            ['node', str(ROOT / 'tests/fixtures' / fixture)],
            stdout=subprocess.PIPE, text=True,
        )

        def stop_fixture():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            proc.stdout.close()

        self.addCleanup(stop_fixture)
        ready = Queue()
        threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
        try:
            port = int(ready.get(timeout=10))
        except (Empty, ValueError):
            self.fail(f'Private HTTP fixture could not start; process exit: {proc.poll()}')
        return f'http://127.0.0.1:{port}'

    def test_connected_report_uses_the_private_http_handler_and_preserves_the_upload(self):
        faulthandler.dump_traceback_later(45)
        self.addCleanup(faulthandler.cancel_dump_traceback_later)
        origin = self.start_node_fixture('private_billing_browser.cjs')
        # No request interception on this independent loopback-only page. Log
        # unexpected origins without holding network traffic during assertions.
        context = self.browser.new_context(locale='en-US')
        self.addCleanup(context.close)
        page = context.new_page()
        page.set_default_timeout(10_000)
        errors, external = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: external.append(request.url) if
                urlparse(request.url).netloc != urlparse(origin).netloc else None)
        print('private browser fixture ready', flush=True)
        page.goto(origin + '/import.html')
        page.locator('#billing-files').set_input_files(ROOT / 'tests/fixtures/cur/synthetic.csv')
        page.locator('#read-files').click()
        expect(page.locator('#step-2')).to_be_visible()
        page.locator('#validate').click()
        expect(page.locator('#step-3')).to_be_visible()
        page.locator('#acknowledge').check()
        page.locator('#open-report').click()
        expect(page.locator('#kpi-total')).to_have_text('$100.00')
        print('private browser original upload verified', flush=True)

        page.goto(origin + '/__fixture/private-browser')
        state = page.locator('#fixture-output')
        headers = page.locator('#fixture-response')
        connected = page.get_by_role('button', name='Open connected report')

        def open_connected(token, status, view_state, evidence):
            print(f'private browser setting fixture session for HTTP {status}', flush=True)
            context.add_cookies([{
                'name': 'atlas-fixture-session', 'value': token, 'url': origin, 'httpOnly': True,
            }])
            connected.click()
            expect(headers).to_have_text(f'{status}|private, no-store|{evidence}')
            expect(state).to_have_text(view_state)

        open_connected('fx-965ed1', 200, 'connected:105', '2026-02-02T02:00:00.000Z|35')
        open_connected('fx-ab64d3', 404, 'unavailable', 'no report')
        open_connected('fx-1186c0', 401, 'unavailable', 'no report')
        page.get_by_role('button', name='Open local report').click()
        expect(state).to_have_text('upload:100')
        page.goto(origin + '/dashboard.html?source=import')
        expect(page.locator('#kpi-total')).to_have_text('$100.00')
        self.assertEqual(errors, [])
        self.assertEqual(external, [])
        print('private browser response and preserved upload verified', flush=True)

    def test_private_refresh_and_disconnect_keep_the_accepted_browser_report(self):
        origin = self.start_node_fixture('private_billing_actions_browser.cjs')
        context = self.browser.new_context(locale='en-US')
        self.addCleanup(context.close)
        page = context.new_page()
        page.set_default_timeout(10_000)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def login(value='fx-operator'):
            context.add_cookies([{
                'name': 'atlas-action-session', 'value': value, 'url': origin, 'httpOnly': True,
            }])

        def control(command):
            request = Request(origin + '/__control?command=' + command, method='POST',
                              headers={'X-Atlas-Fixture-Control': 'yes'})
            with urlopen(request, timeout=8) as response:
                return json.load(response)

        def click(id):
            page.locator('#' + id).click()

        def expect_view(status, accepted=''):
            expect(page.locator('#status')).to_have_text(status)
            expect(page.locator('#accepted')).to_have_text(accepted)

        t1 = '100 | 2026-02-02T01:00:00.000Z'
        t2 = '105 | 2026-02-02T02:00:00.000Z'
        login()
        page.goto(origin)
        click('load')
        expect_view('idle', t1)

        click('refresh')
        expect_view('retry-submit', t1)
        first = control('state')['queued']
        self.assertTrue(first)
        click('refresh')
        expect_view('queued', t1)
        self.assertEqual(control('state')['queued'], first)
        self.assertEqual(control('state')['submits'], 2)
        control('start')
        click('check')
        expect_view('running', t1)
        self.assertEqual(control('finish')['status'], 'completed')
        click('check')
        expect_view('completed', t2)

        control('viewer')
        click('refresh')
        expect_view('read-only', t2)
        expect(page.locator('#refresh')).to_be_disabled()
        expect(page.locator('#disconnect')).to_be_disabled()
        control('manager')
        click('load')
        expect_view('idle', t2)
        control('advance-generation')
        click('disconnect')
        expect_view('selection-reloaded', t2)
        self.assertEqual(control('state')['disconnects'], 1)

        click('refresh')
        expect_view('queued', t2)
        control('start')
        click('check')
        expect_view('running', t2)
        click('disconnect')
        expect_view('disconnected', t2)
        self.assertEqual(control('finish')['status'], 'canceled')
        expect_view('disconnected', t2)
        page.reload()
        click('load')
        expect_view('read-only', t2)
        expect(page.locator('#refresh')).to_be_disabled()
        expect(page.locator('#disconnect')).to_be_disabled()
        control('revoke-read')
        click('load')
        expect_view('hidden')
        expect(page.locator('#refresh')).to_be_disabled()
        login('fx-foreign')
        click('load')
        expect_view('hidden')
        login('fx-revoked')
        click('load')
        expect_view('hidden')
        expect(page.locator('#check')).to_be_disabled()
        self.assertEqual(errors, [])

    def test_private_drilldown_discards_stale_and_delayed_success(self):
        origin = self.start_node_fixture('private_billing_actions_browser.cjs')
        context = self.browser.new_context(locale='en-US')
        self.addCleanup(context.close)
        page = context.new_page()
        page.set_default_timeout(10_000)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def cookie(value):
            context.add_cookies([{
                'name': 'atlas-action-session', 'value': value, 'url': origin, 'httpOnly': True,
            }])

        def publish(command):
            request = Request(origin + '/__control?command=' + command, method='POST',
                              headers={'X-Atlas-Fixture-Control': 'yes'})
            with urlopen(request, timeout=8) as response:
                self.assertEqual(json.load(response)['status'], 'completed')

        def click(id):
            page.locator('#' + id).click()

        def view(state, total='', rows=''):
            expect(page.locator('#view-status')).to_have_text(state)
            expect(page.locator('#total')).to_have_text(total)
            expect(page.locator('#rows')).to_have_text(rows)

        cookie('fx-operator')
        page.goto(origin + '/detail-view')
        click('overview-load')
        view('ready', '100')
        publish('correction')
        # The request reaches the server after the correction. Its 409 triggers
        # a fresh overview without applying a detail from the old revision.
        click('detail-load')
        view('ready', '105')
        click('detail-load')
        view('detail-ready', '105', '70,35')

        # This time the old request already succeeded before publication.
        click('hold')
        click('detail-load')
        expect(page.locator('#held')).to_have_text('held')
        publish('newer')
        click('overview-load')
        view('ready', '110')
        click('detail-load')
        view('detail-ready', '110', '70,40')
        expect(page.locator('#asked-offset')).to_have_text('0')
        click('release')
        expect(page.locator('#held')).to_have_text('released')
        view('detail-ready', '110', '70,40')

        cookie('fx-revoked')
        click('detail-load')
        view('hidden')
        expect(page.locator('#detail-load')).to_be_disabled()
        self.assertEqual(errors, [])

    def test_samples_save_reopen_download_demo_isolation_and_clear(self):
        self.choose([ROOT / 'mock_aws_cur_iaas_heavy_named20251122.csv', ROOT / 'mock_gcp_billing.csv'])
        expect(self.page.locator('#format-0')).to_have_value('aws')
        expect(self.page.locator('#format-1')).to_have_value('gcp')
        self.page.locator('#currency-assumption').fill('USD')
        self.validate()
        expect(self.page.locator('#preview-metrics')).to_contain_text('$233,951.95')
        expect(self.page.locator('#validation-summary')).to_contain_text('1,203')
        self.page.screenshot(path=str(self.artifacts / 'import-preview-desktop.png'), full_page=True)
        self.accept()
        expect(self.page.locator('#kpi-total')).to_have_text('$233,951.95')
        expect(self.page.locator('#kpi-wasted')).to_have_text('Unavailable')
        expect(self.page.locator('#iac-download')).to_be_disabled()
        expect(self.page.locator('#chart-status')).to_contain_text('Cost tables and analysis remain available')
        expect(self.page.locator('#import-status')).to_contain_text('saved only in this browser')
        self.page.reload()
        expect(self.page.locator('#kpi-total')).to_have_text('$233,951.95')
        self.page.locator('.report-actions summary').click()
        with self.page.expect_download() as download_info:
            self.page.locator('#download-imported-report').click()
        report = json.loads(Path(download_info.value.path()).read_text())
        self.assertFalse(report['sampleData'])
        self.assertEqual(report['analysis']['rowCount'], 1203)
        self.assertEqual(report['analysis']['importMethod'], 'browser')
        self.assertAlmostEqual(report['analysis']['reconciliation']['sourceTotal'], 233951.9462)
        self.assertAlmostEqual(report['analysis']['reconciliation']['difference'], 0.0038)
        self.page.goto(self.origin + '/dashboard_demo.html')
        expect(self.page.locator('#mode-badge-header')).to_contain_text('DEMO DATA')
        self.page.goto(self.origin + '/dashboard.html')
        expect(self.page.locator('#kpi-total')).to_have_text('$233,951.95')
        self.page.locator('.report-actions summary').click()
        self.page.locator('#clear-imported-report').click()
        self.page.wait_for_url('**/import.html')
        self.assertIsNone(self.page.evaluate('ReportStore.get()'))
        self.page.goto(self.origin + '/dashboard.html?source=import')
        expect(self.page.locator('main')).to_contain_text('Your imported report is not available')

    def test_custom_mapping_safe_labels_and_stale_preview(self):
        dangerous = '<img src=x onerror="window.injected=true">'
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(['Charge', 'Owner', 'Group', 'Currency', 'Service'])
        writer.writerow(['12.34', dangerous, 'core', 'EUR', 'Compute Engine'])
        self.choose([{'name': 'custom.csv', 'mimeType': 'text/csv', 'buffer': text.getvalue().encode()}])
        self.validate()
        expect(self.page.locator('#validation-errors')).to_contain_text('Choose the column containing the cost amount')
        expect(self.page.locator('#open-report')).to_be_disabled()
        self.page.locator('#back-mappings').click()
        self.page.locator('#map-0-cost').select_option('Charge')
        self.page.locator('#map-0-product').select_option('Owner')
        self.page.locator('#map-0-platform').select_option('Group')
        self.page.locator('#provider-0').select_option('gcp')
        self.validate()
        expect(self.page.locator('#preview-metrics')).to_contain_text('€12.34')
        self.page.locator('#acknowledge').check()
        self.page.locator('#back-mappings').click()
        self.page.locator('#map-0-cost').select_option('Owner')
        self.validate()
        expect(self.page.locator('#open-report')).to_be_disabled()
        expect(self.page.locator('#error-rows')).to_contain_text('finite numeric amount')
        self.page.locator('#back-mappings').click()
        self.page.locator('#map-0-cost').select_option('Charge')
        self.validate()
        self.accept()
        self.page.locator('nav a').filter(has_text='Product').click()
        expect(self.page.locator('#table-products')).to_contain_text(dangerous)
        self.assertEqual(self.page.locator('#table-products img').count(), 0)
        self.assertFalse(self.page.evaluate('Boolean(window.injected)'))

    def test_invalid_import_preserves_previous_report_and_duplicate_rejection(self):
        self.page.evaluate('ReportStore.save({marker: "previous-report"})')
        self.choose([{'name':'mixed.csv', 'mimeType':'text/csv', 'buffer':b'Cost,Currency\n1,USD\n2,EUR'}])
        self.validate()
        expect(self.page.locator('#error-rows')).to_contain_text('Cannot combine USD, EUR')
        expect(self.page.locator('#download-report')).to_be_disabled()
        self.assertEqual(self.page.evaluate('ReportStore.get()'), {'marker':'previous-report'})
        self.page.goto(self.origin + '/import.html')
        payload = b'Cost,Currency\n1,USD'
        self.choose([{'name':name, 'mimeType':'text/csv', 'buffer':payload} for name in ['first.csv','renamed.csv']])
        self.validate()
        expect(self.page.locator('#error-rows')).to_contain_text('identical contents')
        expect(self.page.locator('#open-report')).to_be_disabled()

    def test_mobile_mapping_preview_and_storage_failure_keeps_download(self):
        self.page.set_viewport_size({'width':390,'height':844})
        self.choose([{'name':'mobile.csv','mimeType':'text/csv','buffer':b'Cost,Product,Currency\n42,api,USD'}])
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.page.screenshot(path=str(self.artifacts / 'import-mapping-mobile.png'), full_page=True)
        self.validate()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.page.screenshot(path=str(self.artifacts / 'import-preview-mobile.png'), full_page=True)
        self.page.locator('#acknowledge').check()
        self.page.evaluate('() => { ReportStore.save = async () => { throw new Error("Browser storage is full"); }; }')
        self.page.locator('#open-report').click()
        expect(self.page.locator('#error')).to_contain_text('Browser storage is full')
        expect(self.page.locator('#download-report')).to_be_enabled()
        with self.page.expect_download() as download_info:
            self.page.locator('#download-report').click()
        self.assertEqual(json.loads(Path(download_info.value.path()).read_text())['TotalBilledCost'], 42)

    def test_file_errors_and_remove_retry(self):
        self.page.locator('#billing-files').set_input_files({'name':'broken.csv','mimeType':'text/csv','buffer':b'Cost,Product\n1,"unterminated'})
        self.page.locator('#read-files').click()
        expect(self.page.locator('#error')).to_contain_text('closing quote')
        expect(self.page.locator('#step-1')).to_be_visible()
        self.page.get_by_role('button',name='Remove broken.csv').click()
        expect(self.page.locator('#read-files')).to_be_disabled()
        self.choose([{'name':'fixed.csv','mimeType':'text/csv','buffer':b'Cost,Product\n1,app'}])
        self.validate()
        expect(self.page.locator('#validation-badge')).to_have_text('READY TO REVIEW')


if __name__ == '__main__':
    unittest.main(verbosity=2)
