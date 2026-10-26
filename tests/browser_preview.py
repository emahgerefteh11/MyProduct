"""Preview acceptance: real uploads, isolation, staged adoption, and mobile UX."""
import json
from urllib.parse import urlparse
import unittest
import zipfile

from playwright.sync_api import expect
import browser_dashboard as dashboard_tests
from browser_import import ROOT


class BrowserPreview(unittest.TestCase):
    setUpClass = classmethod(dashboard_tests.BrowserDashboard.setUpClass.__func__)
    tearDownClass = classmethod(dashboard_tests.BrowserDashboard.tearDownClass.__func__)

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US', reduced_motion='reduce')
        self.context.route('**/*', lambda route: route.continue_() if urlparse(route.request.url).netloc == urlparse(self.origin).netloc else route.abort())
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], 'Preview JavaScript errors')

    def goto(self, path):
        self.page.goto(self.origin + '/preview/' + path)

    def upload(self, files):
        self.goto('import.html')
        self.page.locator('#billing-files').set_input_files(files)
        self.page.locator('#read-files').click()
        expect(self.page.locator('#step-2')).to_be_visible()
        self.page.locator('#validate').click()
        expect(self.page.locator('#valid-preview')).to_be_visible()
        expect(self.page.locator('#status')).to_be_hidden()
        if self.page.locator('#acknowledge').is_visible():
            self.page.locator('#acknowledge').check()
        self.page.locator('#open-report').click()
        self.page.wait_for_url('**/preview/dashboard.html?source=import')
        expect(self.page.locator('#main-content')).to_have_attribute('aria-busy', 'false')

    def test_layout_and_spend_hierarchy_on_desktop_and_mobile(self):
        for width in (1440, 768, 390, 320):
            self.page.set_viewport_size({'width': width, 'height': 1000 if width == 1440 else 844})
            for name, path in [('home', ''), ('dashboard', 'dashboard_demo.html'), ('import', 'import.html'), ('adoption', 'adoption.html')]:
                with self.subTest(width=width, page=name):
                    self.goto(path)
                    if name == 'dashboard':
                        expect(self.page.locator('#main-content')).to_have_attribute('aria-busy', 'false')
                        expect(self.page.locator('#preview-driver')).to_have_text('Storage accounts for 69.5%')
                        expect(self.page.locator('#preview-period')).to_contain_text('Billing period not supplied')
                        if width < 700:
                            self.assertLess(self.page.locator('#kpi-total').bounding_box()['y'], 320)
                    self.assertLessEqual(self.page.evaluate('document.documentElement.scrollWidth'), width, f'{name} overflows at {width}px')
                    expect(self.page.locator('.preview-banner')).to_be_visible()
                    if width in (1440, 390):
                        self.page.screenshot(path=str(self.artifacts / f'preview-{name}-{width}.png'), full_page=True, animations='disabled')

    def test_filters_products_and_view_history_preserve_context(self):
        self.goto('dashboard_demo.html')
        expect(self.page.locator('#kpi-total')).to_have_text('$133,951.95')
        self.page.locator('#chart-platform .rank-row').first.click()
        expect(self.page.locator('#kpi-total')).to_have_text('$32,222.65')
        expect(self.page.locator('#preview-driver-detail')).to_contain_text('PlatformE')
        self.page.locator('[data-view="products"]').click()
        expect(self.page.locator('#spend-hero')).to_be_hidden()
        self.page.locator('#table-products .table-link').first.click()
        expect(self.page.locator('#product-dialog-title')).to_have_text('MarketingSite')
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#table-products .table-link').first).to_be_focused()
        self.page.locator('[data-view="iac"]').click()
        expect(self.page.locator('#page-title')).to_have_text('Manage as code')
        self.page.go_back()
        expect(self.page.locator('#tab-products')).to_be_visible()
        self.page.locator('[data-view="overview"]').click()
        expect(self.page.locator('#spend-hero')).to_be_visible()
        expect(self.page.locator('#kpi-total')).to_have_text('$32,222.65')

    def test_import_separators_are_detected_per_file_and_storage_is_isolated(self):
        self.page.goto(self.origin + '/import.html')
        self.page.evaluate('value => ReportStore.save(value)', dashboard_tests.report_fixture())
        self.goto('import.html')
        self.assertIsNone(self.page.evaluate('() => ReportStore.get()'))
        semicolon = 'BilledCost;BillingCurrency;ServiceCategory;ProviderName;"Note,;"\n10;USD;Object Storage;aws;example\n'
        tab = 'lineItem/UnblendedCost\tlineItem/CurrencyCode\tlineItem/ProductCode\n5\tUSD\tAmazonEC2\n'
        self.upload([{'name': 'focus.csv', 'mimeType': 'text/csv', 'buffer': semicolon.encode()}, {'name': 'aws.csv', 'mimeType': 'text/csv', 'buffer': tab.encode()}])
        expect(self.page.locator('#kpi-total')).to_have_text('$15.00')
        report = self.page.evaluate('() => ReportStore.get()')
        self.assertEqual([source['delimiter'] for source in report['analysis']['sources']], [';', '\t'])
        self.page.goto(self.origin + '/dashboard.html?source=import')
        expect(self.page.locator('#kpi-total')).to_have_text('$150.00')
        self.goto('dashboard.html?source=import')
        expect(self.page.locator('#kpi-total')).to_have_text('$15.00')
        self.goto('')
        expect(self.page.locator('#return-report')).to_be_visible()
        self.goto('dashboard_demo.html')
        expect(self.page.locator('#kpi-total')).to_have_text('$133,951.95')

    def test_optional_mappings_and_override_remain_available(self):
        self.goto('import.html')
        expect(self.page.locator('#delimiter')).to_be_hidden()
        self.page.locator('.advanced-options summary').click()
        self.page.locator('#delimiter').select_option(';')
        self.page.locator('#billing-files').set_input_files({'name': 'custom.csv', 'mimeType': 'text/csv', 'buffer': b'amount;currency;service;resource\n8;USD;Compute;instance-a\n'})
        self.page.locator('#read-files').click()
        expect(self.page.locator('#step-2')).to_be_visible()
        expect(self.page.locator('#map-0-cost')).to_be_visible()
        expect(self.page.locator('#map-0-resource')).to_be_hidden()
        self.page.locator('.optional-mappings summary').click()
        expect(self.page.locator('#map-0-resource')).to_be_visible()
        self.page.locator('#map-0-cost').select_option('amount')
        self.page.locator('#validate').click()
        expect(self.page.locator('#valid-preview')).to_be_visible()

    def test_commands_appear_after_successful_package_and_tool_download_is_valid(self):
        csv = 'lineItem/UnblendedCost,lineItem/ResourceId,lineItem/UsageAccountId,product/regionCode,lineItem/ProductCode,lineItem/CurrencyCode\n10,i-0123456789abcdef0,123456789012,us-east-1,AmazonEC2,USD\n4,vol-0123456789abcdef0,123456789012,us-east-1,AmazonEC2,USD\n'
        self.upload({'name': 'aws.csv', 'mimeType': 'text/csv', 'buffer': csv.encode()})
        self.goto('adoption.html')
        expect(self.page.locator('#workspace')).to_be_visible()
        expect(self.page.locator('#local-steps')).to_be_hidden()
        self.page.locator('#select-ready').click()
        expect(self.page.locator('#download-package')).to_be_enabled()
        expect(self.page.locator('#local-steps')).to_be_hidden()
        with self.page.expect_download() as download:
            self.page.locator('#download-package').click()
        with zipfile.ZipFile(download.value.path()) as package:
            self.assertIsNone(package.testzip())
            self.assertEqual(package.read('adoption_tool.py'), (ROOT / 'adoption_tool.py').read_bytes())
            self.assertEqual(len(json.loads(package.read('candidates.json'))['resources']), 2)
        expect(self.page.locator('#local-steps')).to_be_visible()
        expect(self.page.locator('#generate-title')).to_be_focused()
        self.assertEqual(self.page.locator('.steps [aria-current="step"]').inner_text(), '2\nGenerate & review locally')
        self.context.grant_permissions(['clipboard-read', 'clipboard-write'], origin=self.origin)
        self.page.locator('.command-copy').first.click()
        expect(self.page.locator('.copy-status').first).to_have_text('Commands copied.')
        self.assertIn('python adoption_tool.py discover', self.page.evaluate('() => navigator.clipboard.readText()'))
        self.page.screenshot(path=str(self.artifacts / 'preview-adoption-package.png'), full_page=True)
        self.goto('adoption.html')
        expect(self.page.locator('#local-steps')).to_be_hidden()
        self.page.locator('.return-results summary').click()
        expect(self.page.locator('#inventory-file')).to_be_visible()

    def test_credit_adjustments_and_data_warnings_stay_explicit(self):
        value = dashboard_tests.report_fixture()
        value['TotalBilledCost'] = 50
        value['services'] = [{'ServiceCategory': 'Object Storage', 'BilledCost': 100}, {'ServiceCategory': 'Compute/VM', 'BilledCost': -50}]
        value['analysis']['currency'] = None
        value['analysis']['usagePeriod']['complete'] = False
        value['analysis']['warnings'] = ['Some rows have no product allocation.']
        self.goto('import.html')
        self.page.evaluate('value => ReportStore.save(value)', value)
        self.goto('dashboard.html?source=import')
        expect(self.page.locator('#preview-driver')).to_have_text('Storage is the largest charge')
        expect(self.page.locator('#preview-driver-detail')).to_contain_text('credits or adjustments')
        expect(self.page.locator('#preview-alert')).to_contain_text('Some rows have no product allocation.')
        expect(self.page.locator('#preview-period')).to_contain_text('Currency not supplied')
        self.page.locator('#preview-alert a').click()
        self.assertTrue(self.page.locator('.quality-panel details').evaluate('(el) => el.open'))
        expect(self.page.locator('.quality-panel summary')).to_be_focused()

    def test_motion_is_brief_and_reduced_motion_is_respected(self):
        self.page.emulate_media(reduced_motion='no-preference')
        self.goto('')
        active = self.page.evaluate('() => document.getAnimations().map(a => a.effect.getTiming()).map(t => Number(t.duration) + Number(t.delay))')
        self.assertTrue(active)
        self.assertLessEqual(max(active), 700)
        self.page.emulate_media(reduced_motion='reduce')
        self.page.wait_for_function("document.getAnimations().every(a => a.playState !== 'running')")
        self.assertEqual(self.page.locator('.hero-copy').evaluate('(el) => getComputedStyle(el).opacity'), '1')


if __name__ == '__main__':
    unittest.main()
