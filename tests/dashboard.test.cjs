// Exercise the actual dashboard script with a minimal DOM, without network/CDN access.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const fixture = JSON.parse(fs.readFileSync(path.join(root, 'report_data.json'), 'utf8'));
delete fixture.sampleData; // Exercise backward-compatible loading of a legacy report.

class Element {
    constructor(id = '') {
        this.id = id;
        this.children = [];
        this.style = {};
        this.dataset = {};
        this.classList = {add() {}, remove() {}};
        this.selectedIndex = 0;
        this._value = undefined;
        this._html = '';
        this._text = '';
    }
    set innerHTML(value) { this._html = value; this._text = ''; this.children = []; }
    get innerHTML() { return this._html; }
    set textContent(value) { this._text = String(value); this._html = ''; this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    set value(value) { this._value = value; }
    get value() { return this._value ?? this.children[this.selectedIndex]?.value ?? ''; }
    get options() { return this.children; }
    appendChild(child) { this.children.push(child); return child; }
}

async function dashboard({data = fixture, demo = false, available = true, stored = null} = {}) {
    const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element(match[1])]));
    const tables = new Map();
    const main = new Element('main');
    const document = {
        getElementById(id) { assert.ok(elements.has(id), `DOM target exists: ${id}`); return elements.get(id); },
        createElement() { return new Element(); },
        querySelector(selector) {
            if (selector === 'main') return main;
            const table = selector.match(/^#(.+) tbody$/);
            assert.ok(table && elements.has(table[1]), `Table target exists: ${selector}`);
            if (!tables.has(selector)) tables.set(selector, new Element());
            return tables.get(selector);
        },
        querySelectorAll() { return []; },
    };
    elements.get('iac-framework').value = 'terraform';
    elements.get('iac-env').value = 'dev';
    const context = vm.createContext({
        document, console, Intl, Date, setTimeout,
        fetch: async () => ({ok: available, json: async () => structuredClone(data)}),
        ReportStore: {get: async () => structuredClone(stored)},
    });
    context.window = context;
    const prefix = demo ? `const embeddedReportData = ${JSON.stringify(data)};\n` : '';
    await vm.runInContext(prefix + script, context);
    assert.doesNotMatch(main.innerHTML, /Error Loading Data/);
    return {context, elements, tables, main, run: code => vm.runInContext(code, context)};
}

test('report mode keeps costs but never manufactures optimization evidence', async () => {
    const page = await dashboard();
    assert.equal(page.elements.get('kpi-total').textContent, '133,951.95');
    assert.equal(page.elements.get('kpi-wasted').textContent, 'Unavailable');
    assert.equal(page.elements.get('kpi-mom-growth').textContent, 'Period comparison unavailable');
    assert.equal(page.elements.get('finops-total-savings').textContent, 'Unavailable');
    assert.equal(page.context.finopsRecommendations.length, 0);
    assert.match(page.tables.get('#table-recommendations tbody').textContent, /No verified findings/);
    assert.doesNotMatch(page.tables.get('#table-ri-coverage tbody').innerHTML, /Purchase Savings Plan/);
    assert.ok(page.tables.get('#table-products tbody').children.length > 0);
    assert.match(page.elements.get('mode-badge-header').innerHTML, /REPORT DATA/);
});

test('old reports never claim a fresh analysis date or assumed USD', async () => {
    const page = await dashboard();
    assert.match(page.elements.get('last-updated').textContent, /Analysis time unavailable/);
    assert.match(page.elements.get('report-provenance').textContent, /no source, date, or currency metadata/);
    assert.equal(page.run('fmtCurrency(100)'), '100.00');
});

test('the bundled sample is labeled as demo even when fetched by the report page', async () => {
    const data = JSON.parse(fs.readFileSync(path.join(root, 'report_data.json'), 'utf8'));
    assert.equal(data.sampleData, true);
    const page = await dashboard({data});
    assert.match(page.elements.get('mode-badge-header').innerHTML, /DEMO DATA/);
    assert.match(page.elements.get('evidence-notice').textContent, /simulated examples/);
});

test('currency, timestamps, provenance warnings, and cost basis come from the report', async () => {
    const data = structuredClone(fixture);
    data.analysis = {
        generatedAt: '2025-01-02T03:04:05Z', currency: 'EUR', costBasis: 'effective',
        sourceCount: 1, rowCount: 1200,
        sources: [{file: '<img src=x onerror=alert(1)>', schema: 'focus', rows: 1200}],
        usagePeriod: {start: '2025-01-01', end: null, complete: false},
        reconciliation: {sourceTotal: 133951.95, reportTotal: 133951.95, difference: 0},
        warnings: ['<script>this must remain text</script>'],
    };
    const page = await dashboard({data});
    assert.match(page.elements.get('kpi-total').textContent, /€/);
    assert.match(page.elements.get('last-updated').textContent, /2025/);
    assert.equal(page.elements.get('kpi-total-label').textContent, 'Total Effective Cost');
    assert.match(page.elements.get('report-provenance').textContent, /incomplete/);
    assert.equal(page.elements.get('report-provenance').innerHTML, '');
    assert.equal(page.elements.get('report-warnings').children[0].innerHTML, '');
});

test('demo examples are labeled and cannot leak when returning to report mode', async () => {
    const page = await dashboard({demo: true});
    assert.ok(page.context.finopsRecommendations.length > 0);
    assert.match(page.elements.get('evidence-notice').textContent, /simulated examples/);
    assert.match(page.elements.get('kpi-mom-growth').textContent, /simulated/);
    assert.match(page.elements.get('coverage-subtitle').textContent, /not measured/);
    page.run("setMode('live'); renderDashboard('all')");
    assert.equal(page.context.finopsRecommendations.length, 0);
    assert.equal(page.elements.get('finops-total-savings').textContent, 'Unavailable');
    page.run("filterFinops('storage')");
    assert.match(page.tables.get('#table-recommendations tbody').textContent, /No verified findings/);
});

test('cost tables and IaC preview survive an unavailable chart CDN', async () => {
    const page = await dashboard();
    assert.equal(page.elements.get('chart-status').hidden, false);
    assert.match(page.elements.get('chart-status').textContent, /Cost tables and analysis remain available/);
    assert.ok(page.tables.get('#table-services tbody').children.length > 0);
    assert.match(page.elements.get('iac-preview').textContent, /terraform/);
});

test('unknown architecture is not replaced with a hashed live pattern', async () => {
    const data = structuredClone(fixture);
    for (const platform of data.platforms) {
        for (const product of platform.products) product.services = [];
    }
    const page = await dashboard({data});
    assert.equal(page.elements.get('iac-pattern-badge').style.display, 'none');
    assert.doesNotMatch(page.elements.get('iac-preview').textContent, /module "eks"|module "db"/);
});

test('IaC uses source provider attribution and blocks ambiguous provider guesses', async () => {
    const data = structuredClone(fixture);
    for (const platform of data.platforms) {
        for (const product of platform.products) product.providers = [{ProviderName: 'gcp', BilledCost: product.BilledCost}];
    }
    const page = await dashboard({data});
    assert.match(page.elements.get('iac-preview').textContent, /google_compute_instance/);
    page.run("window.allProductsData.forEach(p => p.data.providers.push({ProviderName: 'aws'})); updateIacPreview()");
    assert.match(page.elements.get('iac-preview').textContent, /multiple or unknown providers/);
    assert.equal(page.elements.get('iac-download').disabled, true);
    assert.equal(page.elements.get('iac-copy').disabled, true);
});

test('missing report produces the existing actionable empty state', async () => {
    const page = await dashboard({available: false});
    assert.match(page.main.innerHTML, /No Analysis Data Yet/);
    assert.match(page.main.innerHTML, /cur_to_json.py/);
});

test('generated demo uses the current dashboard and explicitly embeds data', () => {
    const demo = fs.readFileSync(path.join(root, 'dashboard_demo.html'), 'utf8');
    assert.ok(demo.includes('const embeddedReportData = '));
    assert.ok(demo.includes(script.slice(script.indexOf('let reportData = null;'))));
});

test('accepted browser report overrides bundled sample and keeps unsupported sizing unavailable', async () => {
    const importer = require('../billing_import.js');
    const parsed = importer.parseCSV('Cost,Product,Platform,Service,Currency\n42,<img src=x onerror=alert(1)>,core,Compute Engine,EUR');
    const result = importer.validate([{...parsed, name:'import.csv', sha256:'abc', delimiter:','}], [importer.detect(parsed.headers)]);
    const page = await dashboard({data:{...fixture,sampleData:true},stored:result.report});
    assert.match(page.elements.get('kpi-total').textContent, /42\.00/);
    assert.match(page.elements.get('mode-badge-header').innerHTML, /REPORT DATA/);
    assert.equal(page.elements.get('download-imported-report').hidden, false);
    assert.match(page.tables.get('#table-compute-skus tbody').innerHTML, /not analyzed/);
    assert.equal(page.elements.get('iac-download').disabled, true);
    assert.match(page.tables.get('#table-products tbody').children[0].innerHTML, /&lt;img/);
    assert.doesNotMatch(page.tables.get('#table-products tbody').children[0].innerHTML, /<img/);
    const demo = await dashboard({demo:true,stored:result.report});
    assert.match(demo.elements.get('mode-badge-header').innerHTML, /DEMO DATA/);
});
