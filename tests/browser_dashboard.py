"""Browser checks for the modern workspace, with all third-party traffic blocked."""
import functools
import json
import os
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
import unittest

from playwright.sync_api import sync_playwright, expect
from browser_import import ROOT, QuietHandler


def report_fixture():
    def product(name, cost, provider):
        return {"Product": name, "BilledCost": cost, "environments": [{"Environment": "prod"}], "services": [{"ServiceCategory": "Compute/VM", "BilledCost": cost}], "providers": [{"ProviderName": provider, "BilledCost": cost}]}
    return {
        "sampleData": False, "TotalBilledCost": 150,
        "analysis": {"generatedAt": "2026-09-15T10:00:00Z", "importMethod": "browser", "currency": "USD", "costBasis": "billed", "sourceCount": 1, "rowCount": 3,
                     "sources": [{"file": "billing.csv", "schema": "focus", "rows": 3, "sha256": "a" * 64}],
                     "usagePeriod": {"start": "2026-09-01", "end": "2026-09-15", "complete": True}, "warnings": [],
                     "reconciliation": {"sourceTotal": 150, "reportTotal": 150, "difference": 0}},
        "allocation": {"unallocatedProductCost": 0, "unallocatedPlatformCost": 0},
        "products": [product("Alpha", 130, "aws"), product("Beta", 20, "aws")],
        "services": [{"ServiceCategory": "Compute/VM", "BilledCost": 100}, {"ServiceCategory": "Object Storage", "BilledCost": 50}],
        "providers": [{"ProviderName": "aws", "BilledCost": 100}, {"ProviderName": "gcp", "BilledCost": 50}],
        "platforms": [
            {"Platform": "Core", "BilledCost": 100, "products": [product("Alpha", 80, "aws"), product("Beta", 20, "aws")], "services": [{"ServiceCategory": "Compute/VM", "BilledCost": 80}, {"ServiceCategory": "Object Storage", "BilledCost": 20}], "vmSkus": []},
            {"Platform": "Data", "BilledCost": 50, "products": [product("Alpha", 50, "gcp")], "services": [{"ServiceCategory": "Compute/VM", "BilledCost": 20}, {"ServiceCategory": "Object Storage", "BilledCost": 30}], "vmSkus": []},
        ], "resourceNames": [], "namingConvention": {"source": "Explicit billing columns", "mode": "explicit allocation", "pattern": []},
    }


class BrowserDashboard(unittest.TestCase):
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
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='en-US', reduced_motion='reduce')
        self.external_requests, self.js_errors = [], []

        def route(request):
            if urlparse(request.request.url).netloc == urlparse(self.origin).netloc:
                request.continue_()
            else:
                self.external_requests.append(request.request.url)
                request.abort()

        self.context.route('**/*', route)
        self.page = self.context.new_page()
        self.page.on('pageerror', lambda error: self.js_errors.append(str(error)))

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.js_errors, [])
        self.assertEqual(self.external_requests, [], 'UI must work without requesting fonts, chart libraries or other third-party assets')

    def open_report(self, value=None, suffix=''):
        self.page.goto(self.origin + '/import.html')
        self.page.evaluate('value => ReportStore.save(value)', value or report_fixture())
        self.page.goto(self.origin + '/dashboard.html?source=import' + suffix)
        expect(self.page.locator('#report-content')).to_be_visible()
        expect(self.page.locator('#main-content')).to_have_attribute('aria-busy', 'false')

    def assert_no_overflow(self):
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), f'Page overflow at {self.page.viewport_size}')

    def test_demo_charts_render_locally_and_platform_chart_drills_down(self):
        self.page.goto(self.origin + '/dashboard_demo.html')
        expect(self.page.locator('#chart-tco .donut')).to_be_visible()
        expect(self.page.locator('#mode-badge-header')).to_contain_text('DEMO DATA')
        expect(self.page.locator('#evidence-notice')).to_contain_text('simulated examples')
        self.page.screenshot(path=str(self.artifacts / 'workspace-demo-desktop.png'), full_page=True)
        self.open_report()
        expect(self.page.locator('#kpi-total')).to_have_text('$150.00')
        expect(self.page.locator('#kpi-wasted')).to_have_text('Unavailable')
        self.page.locator('#chart-platform').get_by_role('button', name='Explore Core, $100.00').click()
        expect(self.page.locator('#global-platform-filter')).to_have_value('Core')
        expect(self.page.locator('#kpi-total')).to_have_text('$100.00')
        expect(self.page.locator('#table-services')).to_contain_text('$80.00')
        expect(self.page.locator('#scope-summary')).to_contain_text('Sep 15, 2026')
        self.page.locator('#global-platform-filter').select_option('all')
        expect(self.page.locator('#kpi-total')).to_have_text('$150.00')
        self.page.screenshot(path=str(self.artifacts / 'workspace-report-desktop.png'), full_page=True)

    def test_product_search_sort_details_and_history(self):
        self.open_report(suffix='#products')
        expect(self.page.locator('#tab-products')).to_be_visible()
        self.page.locator('#product-sort').select_option('cost-asc')
        expect(self.page.locator('#table-products tbody tr').first).to_contain_text('Beta')
        self.page.locator('#product-search').fill('data')
        expect(self.page.locator('#product-count')).to_contain_text('1 of 3 products')
        self.page.locator('#table-products .table-link').click()
        expect(self.page.locator('#product-dialog')).to_be_visible()
        expect(self.page.locator('#product-dialog-title')).to_have_text('Alpha')
        expect(self.page.locator('#product-dialog-platform')).to_have_text('Platform: Data')
        expect(self.page.locator('#product-dialog-metrics')).to_contain_text('$50.00')
        expect(self.page.locator('#product-dialog-services')).to_contain_text('Compute/VM')
        self.page.screenshot(path=str(self.artifacts / 'workspace-product-detail.png'), full_page=True)
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#product-dialog')).not_to_be_visible()
        expect(self.page.locator('#table-products .table-link')).to_be_focused()
        self.page.locator('#product-search').fill('nothing-matches')
        expect(self.page.locator('#table-products')).to_contain_text('No products match your search')
        self.page.locator('#global-platform-filter').select_option('Core')
        expect(self.page.locator('#table-products')).to_contain_text('No products match your search')
        self.page.locator('#product-search').fill('')
        expect(self.page.locator('#product-count')).to_contain_text('2 of 2 products')
        self.page.locator('[data-view="compute"]').click()
        expect(self.page.locator('#tab-compute')).to_be_visible()
        self.page.go_back()
        expect(self.page.locator('#tab-products')).to_be_visible()
        expect(self.page.locator('[data-view="products"]')).to_have_attribute('aria-current', 'page')

    def test_mobile_navigation_focus_trap_dialog_and_all_views_fit(self):
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.open_report()
        self.assertTrue(self.page.locator('#workspace-sidebar').evaluate('(el) => el.inert'))
        self.page.locator('#nav-toggle').click()
        expect(self.page.locator('#nav-toggle')).to_have_attribute('aria-expanded', 'true')
        expect(self.page.locator('#nav-close')).to_be_focused()
        self.assertTrue(self.page.locator('#main-content').evaluate('(el) => el.inert'))
        self.page.locator('#sidebar-switch-link a').focus()
        self.page.keyboard.press('Tab')
        expect(self.page.locator('.logo')).to_be_focused()
        self.page.keyboard.press('Shift+Tab')
        expect(self.page.locator('#sidebar-switch-link a')).to_be_focused()
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#nav-toggle')).to_be_focused()
        expect(self.page.locator('#nav-toggle')).to_have_attribute('aria-expanded', 'false')
        self.page.locator('#nav-toggle').click()
        self.page.locator('[data-view="products"]').click()
        expect(self.page.locator('#tab-products')).to_be_visible()
        expect(self.page.locator('#page-title')).to_be_focused()
        self.page.locator('#table-products .table-link').first.click()
        expect(self.page.locator('#product-dialog')).to_be_visible()
        self.assert_no_overflow()
        self.page.keyboard.press('Escape')
        for width in [320, 390, 768, 1024, 1440]:
            self.page.set_viewport_size({'width': width, 'height': 900})
            for view in ['overview', 'products', 'finops', 'compute', 'naming', 'iac']:
                self.page.evaluate('view => {location.hash = view}', view)
                expect(self.page.locator('#tab-' + view)).to_be_visible()
                self.assert_no_overflow()
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.page.evaluate("location.hash = 'overview'")
        expect(self.page.locator('#tab-overview')).to_be_visible()
        self.page.screenshot(path=str(self.artifacts / 'workspace-mobile.png'), full_page=True)

    def test_cloud_connection_entrypoint_is_truthful_and_keyboard_accessible(self):
        self.page.goto(self.origin + '/dashboard.html#connections')
        expect(self.page.locator('#cloud-dialog')).to_be_visible()
        expect(self.page.locator('#cloud-dialog')).to_contain_text('Automatic account connections are planned')
        self.assertEqual(self.page.locator('#cloud-dialog input').count(), 0)
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#cloud-dialog')).not_to_be_visible()
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.page.locator('#nav-toggle').click()
        self.page.locator('.sidebar-actions [data-open-cloud]').click()
        expect(self.page.locator('#cloud-dialog')).to_be_visible()
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#nav-toggle')).to_be_focused()

    def test_loading_missing_report_and_invalid_json_states(self):
        pending, release = [], [False]

        def report_route(route):
            if release[0]:
                route.fulfill(status=404, body='missing')
            else:
                pending.append(route)

        self.page.route('**/report_data.json', report_route)
        with self.page.expect_request('**/report_data.json'):
            self.page.goto(self.origin + '/dashboard.html', wait_until='domcontentloaded')
        expect(self.page.locator('#loading-state')).to_be_visible()
        self.assertTrue(pending)
        release[0] = True
        for request in pending:
            request.fulfill(status=404, body='missing')
        expect(self.page.get_by_role('heading', name='No Analysis Data Yet')).to_be_visible()
        expect(self.page.locator('#main-content')).to_have_attribute('aria-busy', 'false')
        self.assert_no_overflow()
        self.page.unroute('**/report_data.json')
        self.page.route('**/report_data.json', lambda route: route.fulfill(status=200, content_type='application/json', body='{'))
        self.page.reload()
        expect(self.page.get_by_role('heading', name='Error Loading Data')).to_be_visible()
        self.assert_no_overflow()

    def test_negative_costs_and_hostile_labels_remain_honest_and_inert(self):
        value = report_fixture()
        value['TotalBilledCost'] = 70
        value['services'] = [{'ServiceCategory': 'Compute/VM', 'BilledCost': 100}, {'ServiceCategory': 'Credits', 'BilledCost': -30}]
        hostile = '<img src=x onerror="window.injected=true">'
        value['platforms'][0]['products'][0]['Product'] = hostile
        value['platforms'][0]['products'][0]['services'][0]['ServiceCategory'] = hostile
        self.open_report(value)
        expect(self.page.locator('#chart-tco .bar-track.signed')).to_have_count(2)
        expect(self.page.locator('#chart-tco')).to_contain_text('-$30.00')
        self.assertEqual(self.page.locator('#chart-tco .donut').count(), 0)
        self.page.locator('[data-view="products"]').click()
        self.page.locator('#table-products .table-link').filter(has_text=hostile).click()
        expect(self.page.locator('#product-dialog-title')).to_have_text(hostile)
        self.assertEqual(self.page.locator('#product-dialog img').count(), 0)
        self.assertFalse(self.page.evaluate('Boolean(window.injected)'))

    def test_home_and_existing_flows_share_responsive_design_and_skip_links(self):
        for path, title in [('index.html', 'A clearer view'), ('import.html', 'From billing files'), ('adoption.html', 'Your infrastructure.')]:
            for width in [320, 390, 768, 1440]:
                self.page.set_viewport_size({'width': width, 'height': 1000})
                self.page.goto(self.origin + '/' + path)
                expect(self.page.locator('h1')).to_contain_text(title)
                self.assert_no_overflow()
            if path == 'index.html':
                self.page.screenshot(path=str(self.artifacts / 'home-desktop.png'), full_page=True)
                self.page.set_viewport_size({'width': 390, 'height': 844})
                self.page.screenshot(path=str(self.artifacts / 'home-mobile.png'), full_page=True)
                self.page.keyboard.press('Tab')
                expect(self.page.locator('.skip-link')).to_be_focused()
                self.page.keyboard.press('Enter')
                expect(self.page.locator('main')).to_be_focused()


if __name__ == '__main__':
    unittest.main(verbosity=2)
