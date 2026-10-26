"""Build the isolated design preview without changing the production pages.

Run python build_preview.py after changes to shared dashboard/import engines.
The preview keeps its own accepted-report database on the same Pages origin.
"""
from pathlib import Path
from generate_dashboard import inject_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'preview'
OUT.mkdir(exist_ok=True)

BANNER = '''<div class="preview-banner"><span><span class="preview-dot" aria-hidden="true"></span>Design preview</span><a href="../">View current site <span aria-hidden="true">→</span></a></div>'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one preview anchor: {old[:90]!r}')
    return text.replace(old, new, 1)


def shared_page(name, page_class):
    text = (ROOT / name).read_text()
    for asset in ('atlas.css', 'dashboard.css', 'import.css', 'adoption.css', 'motion.css', 'dashboard_ui.js', 'billing_import.js', 'adoption.js'):
        text = text.replace(f'="{asset}"', f'="../{asset}"')
    text = text.replace('</head>', '<meta name="robots" content="noindex,nofollow"><link rel="stylesheet" href="preview.css"><script src="preview.js" defer></script></head>')
    text = text.replace('<body>', f'<body class="design-preview {page_class}">{BANNER}')
    text = text.replace('Adopt into IaC', 'Manage as code').replace('IaC Builder', 'Manage as code')
    text = text.replace('Explore IaC adoption', 'Review infrastructure')
    # Use a simple supported arrow in the page markup, leaving scripts intact.
    markup, separator, script = text.partition('<script>')
    return markup.replace('↗', '→') + separator + script


dashboard = shared_page('dashboard.html', 'preview-workspace')
start = dashboard.index('    <div class="workspace-label">')
end = dashboard.index('    <p class="nav-heading">', start)
dashboard = dashboard[:start] + dashboard[end:]
start = dashboard.index('    <div class="sidebar-actions">')
end = dashboard.index('    <div class="sidebar-bottom">', start)
dashboard = dashboard[:start] + dashboard[end:]
dashboard = dashboard.replace('<p class="nav-heading">Analytics</p>', '<p class="nav-heading">Workspace</p>')
dashboard = dashboard.replace('<p class="eyebrow">Clarity across your cloud</p>', '')
dashboard = replace_once(dashboard, '<a class="button" href="adoption.html">Manage as code <span aria-hidden="true">→</span></a>', '')
start = dashboard.index('        <div class="quality-panel">')
end = dashboard.index('        <section id="tab-overview"', start)
quality = dashboard[start:end]
quality = quality.replace('<p id="evidence-notice" class="subtitle"></p>', '')
quality = quality.replace('</details>', '<p id="evidence-notice" class="subtitle"></p><p id="preview-provider-detail" class="subtitle"></p></details>')
dashboard = dashboard[:start] + dashboard[end:]
dashboard = dashboard.replace('        <footer class="workspace-footer">', quality + '        <footer class="workspace-footer">')
start = dashboard.index('            <div class="kpi-grid">')
end = dashboard.index('            <div class="section-grid">', start)
dashboard = dashboard[:start] + dashboard[end:]
hero = '''
        <div id="spend-hero" class="spend-hero">
            <div class="spend-primary"><p class="spend-label" id="kpi-total-label">Reported spend</p><div class="spend-number" id="kpi-total">—</div><p id="preview-period" class="spend-period"></p><p class="spend-scope" id="kpi-providers-count"></p></div>
            <div class="spend-insight"><span class="insight-label">Largest cost driver</span><strong id="preview-driver">Preparing your report</strong><p id="preview-driver-detail"></p><a href="#service-breakdown" class="text-link">See the breakdown <span aria-hidden="true">→</span></a></div>
        </div>
        <p id="preview-alert" class="preview-alert" role="status" hidden></p>
'''
dashboard = replace_once(dashboard, '    <div id="report-content" hidden>', '    <div id="report-content" hidden>' + hero)
dashboard = dashboard.replace('<span class="pill">Cost mix</span>', '').replace('<span class="pill">Drill down →</span>', '').replace('<span class="pill">Cross-cloud view</span>', '')
dashboard = dashboard.replace('<div class="card full-width"><div class="flex-between"><div><h3>Service breakdown</h3>', '<div class="card full-width" id="service-breakdown"><div class="flex-between"><div><h3>Service breakdown</h3>')
dashboard = dashboard.replace('Your ten largest normalized service families in this scope', 'The largest services in your selected scope')
dashboard = dashboard.replace('&#8599;', '&#8594;')
dashboard = dashboard.replace('<th scope="col">Classification</th>', '')
dashboard = dashboard.replace('                <td><span class="badge badge-gray">Normalized</span></td>\n', '')
dashboard = dashboard.replace("colspan=\"4\" class=\"table-empty\">No service costs", "colspan=\"3\" class=\"table-empty\">No service costs")
# Replace illustrative overview KPIs with a single grounded spend narrative.
start = dashboard.index('        // Unit Economics KPI')
end = dashboard.index('        // TCO Categorization', start)
dashboard = dashboard[:start] + dashboard[end:]
dashboard = dashboard.replace("const paths = ['report_data.json', 'MyProduct/report_data.json'];", "const paths = ['../report_data.json'];")
dashboard = dashboard.replace("if (typeof window.refreshAtlas === 'function') window.refreshAtlas();", "if (typeof window.refreshAtlas === 'function') window.refreshAtlas();\n        if (typeof window.refreshPreview === 'function') window.refreshPreview();")
(OUT / 'dashboard.html').write_text('\n'.join(line.rstrip() for line in dashboard.splitlines()) + '\n')
inject_data(str(OUT / 'dashboard.html'), str(ROOT / 'report_data.json'), str(OUT / 'dashboard_demo.html'))

upload = shared_page('import.html', 'preview-flow')
upload = upload.replace('From billing files<br>to a clearer picture.', 'Your cloud costs.<br>Clearly understood.')
upload = upload.replace('Bring your cloud costs together. Review the data before making it your report.', 'Upload a billing export. We’ll match the columns, then help you review the numbers.')
upload = upload.replace('YOUR DATA, YOUR BROWSER', 'Upload billing')
upload = upload.replace('Choose your billing files', 'Start with your billing export')
upload = upload.replace('AWS CUR, Azure Cost Management, Google Cloud, FOCUS, or a custom CSV.', 'AWS CUR, Google Cloud, Azure, and FOCUS CSV exports are supported.')
upload = upload.replace('<option value=",">Comma (,)</option>', '<option value="auto">Detect automatically</option><option value=",">Comma (,)</option>')
upload = replace_once(upload, '<label class="field compact" for="delimiter">Column separator', '<details class="advanced-options"><summary>Advanced file options</summary><label class="field compact" for="delimiter">Column separator')
upload = replace_once(upload, '<option value="tab">Tab</option></select></label>', '<option value="tab">Tab</option></select></label><p class="muted small">Separators are detected for each file. Choose an override if the columns look incorrect.</p></details>')
upload = upload.replace('Files are processed on this device. An accepted report is saved only in this browser when you open the dashboard. You can download it or clear it later.', 'Processed on your device. Preview reports are saved separately from your current dashboard. You can download or clear them at any time.')
upload = upload.replace('Review your column mappings', 'Check the columns we found')
(OUT / 'import.html').write_text(upload)

adoption = shared_page('adoption.html', 'preview-flow preview-adoption')
adoption = adoption.replace('Your infrastructure.<br>Now managed as code.', 'A considered path<br>to infrastructure as code.')
adoption = adoption.replace('BRING EXISTING INFRASTRUCTURE INTO CODE', 'Manage as code')
adoption = adoption.replace('Start with your billing report. Discover current configuration, generate Terraform, and adopt existing resources in reviewed batches.', 'Choose resources from your report, then generate and review Terraform locally with your cloud account.')
adoption = adoption.replace('<section class="panel followup" aria-labelledby="generate-title">', '<section id="local-steps" class="panel followup" aria-labelledby="generate-title" hidden>')
adoption = adoption.replace('<h2 id="generate-title">Generate, review, and adopt</h2>', '<h2 id="generate-title" tabindex="-1">Your package is ready. Review it locally.</h2>')
adoption = adoption.replace('<section id="local-steps"', '<p class="adoption-next"><strong>What happens next</strong> Choose a batch to download your local discovery tool and step-by-step commands. You’ll review the Terraform plan before importing existing resources.</p><section id="local-steps"')
adoption = adoption.replace('Back to dashboard →', 'Back to your report →')
# Returning users can reopen local results without downloading another package.
start = adoption.index('            <h3>Bring discovery and adoption progress back here</h3>')
end = adoption.index('</section>', start)
progress = adoption[start:end].replace('<h3>Bring discovery and adoption progress back here</h3>', '<summary>Have local discovery results? Update your progress.</summary>')
adoption = adoption[:start] + adoption[end:]
adoption = adoption.replace('<noscript>', '<details class="return-results">' + progress + '</details><noscript>')
(OUT / 'adoption.html').write_text(adoption)

store = (ROOT / 'report_store.js').read_text().replace("const DATABASE = 'cloud-cost-atlas'", "const DATABASE = 'cloud-cost-atlas-design-preview'")
(OUT / 'report_store.js').write_text(store)

import_ui = (ROOT / 'import.js').read_text()
# Keep essential cost/allocation fields visible; preserve all other mappings.
import_ui = import_ui.replace("            for (const [key, label] of Object.entries(BillingImport.FIELDS)) {", "            const advanced = node('details', null, 'optional-mappings'); advanced.append(node('summary', 'Resource IDs, dates, and tags (optional)'));\n            const optional = node('div', null, 'mapping-grid');\n            for (const [key, label] of Object.entries(BillingImport.FIELDS)) {")
import_ui = import_ui.replace("selectField(grid, label, `map-${i}-${key}`", "selectField(['cost', 'currency', 'service', 'product', 'platform', 'environment'].includes(key) ? grid : optional, label, `map-${i}-${key}`")
import_ui = import_ui.replace('            section.append(grid);', '            advanced.append(optional); section.append(grid, advanced);')
(OUT / 'import.js').write_text(import_ui)

worker = (ROOT / 'billing_worker.js').read_text().replace("importScripts('billing_import.js')", "importScripts('../billing_import.js')")
worker = worker.replace('let prepared = [];', '''let prepared = [];
// Count only unquoted separators in the first logical CSV record.
function detectDelimiter(text) {
    const counts = {',': 0, ';': 0, '\\t': 0};
    let quoted = false;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (ch === '"') {
            if (quoted && text[i + 1] === '"') i++;
            else quoted = !quoted;
        } else if (!quoted) {
            if (ch === '\\n' || ch === '\\r') break;
            if (ch in counts) counts[ch]++;
        }
    }
    return Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
}''')
worker = worker.replace('const {files, delimiter} = data, {LIMITS} = BillingImport;', 'const {files} = data, {LIMITS} = BillingImport;')
worker = worker.replace('                    const parsed = BillingImport.parseCSV(text, delimiter);', "                    const delimiter = data.delimiter === 'auto' ? detectDelimiter(text) : data.delimiter;\n                    const parsed = BillingImport.parseCSV(text, delimiter);")
(OUT / 'billing_worker.js').write_text(worker)

adoption_ui = (ROOT / 'adoption_ui.js').read_text().replace("fetch('adoption_tool.py'", "fetch('../adoption_tool.py'")
adoption_ui = adoption_ui.replace("anchor.click(); anchor.remove(); setTimeout", "anchor.click(); anchor.remove(); $('local-steps').hidden = false; $('generate-title').focus(); document.querySelectorAll('.steps li').forEach((step, index) => index === 1 ? step.setAttribute('aria-current', 'step') : step.removeAttribute('aria-current')); setTimeout")
adoption_ui = adoption_ui.replace("const matched = items.filter(r => r.discovery);", "const importing = items.some(r => ['imported', 'imported-needs-review', 'import-state-unknown'].includes(r.discovery?.status));\n        document.querySelectorAll('.steps li').forEach((step, index) => index === (importing ? 2 : 1) ? step.setAttribute('aria-current', 'step') : step.removeAttribute('aria-current'));\n        const matched = items.filter(r => r.discovery);")
(OUT / 'adoption_ui.js').write_text(adoption_ui)
print('Preview built at preview/; production pages and storage are unchanged.')
