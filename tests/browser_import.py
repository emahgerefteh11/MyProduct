"""Browser acceptance tests. Run separately after installing Playwright Chromium."""
import csv
import functools
import io
import json
import os
from pathlib import Path
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
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
        self.page.get_by_role('link', name='Adopt into IaC').click()
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
