'use strict';
(() => {
    const $ = id => document.getElementById(id);
    let report = null, items = [], selected = new Set(), inventory = null, busy = false;
    const labels = {candidate: 'Ready for discovery', unresolved: 'Needs review', verified: 'Live resource verified', imported: 'Imported · clean plan', 'imported-needs-review': 'Imported · baseline needs review', 'import-state-unknown': 'Inspect partial import state', 'already-managed': 'Already managed', 'not-found': 'No longer found', error: 'Discovery failed', 'dependency-review': 'Review disk ownership'};
    function node(tag, value, className) { const el = document.createElement(tag); if (value != null) el.textContent = value; if (className) el.className = className; return el; }
    function error(message) { $('error').textContent = message; $('error').hidden = false; $('error').focus(); }
    function money(amount) { return new Intl.NumberFormat(undefined, report?.analysis?.currency ? {style: 'currency', currency: report.analysis.currency} : {minimumFractionDigits: 2, maximumFractionDigits: 2}).format(amount); }
    function controls() { $('download-package').disabled = busy || !selected.size || selected.size > Adoption.MAX_BATCH || !!report?.sampleData; $('selection-summary').textContent = `${selected.size} selected · ${money(items.filter(r => selected.has(Adoption.identity(r))).reduce((n, r) => n + r.cost, 0))} reported cost`; }
    function renderRows() {
        $('resource-rows').replaceChildren();
        const filtered = items.filter(r => $('resource-filter').value === 'all' || r.status === $('resource-filter').value);
        for (const item of filtered.slice(0, 500)) {
            const tr = node('tr'), choice = node('td'), checkbox = node('input'); checkbox.type = 'checkbox'; checkbox.setAttribute('aria-label', `Select ${item.resourceId}`);
            const key = Adoption.identity(item); checkbox.disabled = item.status !== 'candidate' || !!report.sampleData || ['imported', 'imported-needs-review', 'import-state-unknown', 'already-managed'].includes(item.discovery?.status); checkbox.checked = selected.has(key) && !checkbox.disabled;
            checkbox.onchange = () => { if (checkbox.checked) { if (selected.size >= Adoption.MAX_BATCH) { checkbox.checked = false; error('Use at most 200 resources per batch.'); return; } selected.add(key); } else selected.delete(key); controls(); }; choice.append(checkbox);
            const resource = node('td'); resource.append(node('span', item.resourceId || 'Missing resource ID', 'resource-id'), node('span', [item.provider, item.account || 'Account/project missing', item.zone || item.region || 'Region missing'].join(' · '), 'scope-text'));
            const type = node('td'); type.append(node('span', item.resourceType || 'Unresolved'), node('span', `${item.rows.toLocaleString()} billing row(s) · ${item.sourceHashes.length} source(s)`, 'scope-text'));
            const next = node('td'), discovery = item.discovery; const status = discovery?.status || item.status;
            next.append(node('span', labels[status] || status, 'status-label ' + (['candidate', 'verified', 'imported'].includes(status) ? 'good' : 'gap')), node('span', discovery?.message || item.message));
            if (discovery?.checkedAt) next.append(node('span', discovery.checkedAt, 'scope-text'));
            tr.append(choice, resource, type, node('td', money(item.cost)), next); $('resource-rows').append(tr);
        }
        if (!filtered.length) { const td = node('td', 'No resources match this view.'); td.colSpan = 5; const row = node('tr'); row.append(td); $('resource-rows').append(row); }
        $('table-limit').textContent = filtered.length > 500 ? `Showing the first 500 of ${filtered.length.toLocaleString()} resources. Filter to ready resources; work in batches of up to 200.` : `${filtered.length.toLocaleString()} resources in this view. Costs cover the imported period and may include credits.`;
        controls();
    }
    function rebuild() {
        if (!report) return;
        const next = Adoption.candidates(report, {awsAccount: $('aws-account').value.trim(), awsRegion: $('aws-region').value.trim()});
        items = inventory ? Adoption.results(inventory, next) : next;
        selected = new Set();
        $('workspace').hidden = false; $('report-summary').textContent = report.sampleData ? 'Sample report: preview only. Import your own billing data to prepare an adoption package.' : `Billing report generated ${report.analysis?.generatedAt || 'at an unknown time'} · ${report.analysis?.currency || 'Currency unspecified'} · ${report.analysis?.costBasis || 'Unknown cost basis'} costs.`;
        const ready = items.filter(r => r.status === 'candidate'), imported = items.filter(r => r.discovery?.status === 'imported');
        const metrics = [[`${items.length.toLocaleString()} resources`, 'Identified in billing'], [`${ready.length.toLocaleString()} ready`, 'For live discovery'], [`${imported.length.toLocaleString()} imported`, 'Reported by local tool']];
        $('metrics').replaceChildren(...metrics.map(([value, label]) => { const el = node('div', null, 'metric'); el.append(node('span', label), node('strong', value)); return el; }));
        const summary = report.resourceInventorySummary;
        $('coverage-note').textContent = `${items.length - ready.length} resource(s) need scope information or type support. ${summary ? `${summary.unidentifiedRows.toLocaleString()} billing row(s), ${money(summary.unidentifiedCost)}, have no resource ID and remain outside resource adoption.` : 'Billing rows without IDs were not measured by this report version.'} Verified imported resources account for ${money(imported.reduce((n, r) => n + r.cost, 0))} in this billing period. Coverage is based on a local tool result, not an independent audit.`;
        renderRows();
    }
    async function openReport(value) {
        if (!value || typeof value !== 'object' || !Array.isArray(value.resourceInventory)) throw new Error('This JSON lacks scoped resource inventory. Reimport billing CSVs or regenerate the report with the current analyzer.');
        if (value.analysis?.currency && !/^[A-Z]{3}$/.test(value.analysis.currency)) throw new Error('Report currency must be a three-letter code.');
        // Build first, so a rejected file never replaces the current preview.
        Adoption.candidates(value);
        report = value; inventory = null; $('aws-account').value = ''; $('aws-region').value = ''; rebuild();
    }
    async function action(message, task) {
        if (busy) return;
        busy = true; $('error').hidden = true; $('status').textContent = message; $('status').hidden = false; controls();
        try { await task(); } catch (e) { error(e.message); } finally { busy = false; $('status').hidden = true; controls(); }
    }
    async function fileJSON(file) { if (!file || file.size > 10 * 1024 * 1024) throw new Error('Choose a JSON file of at most 10 MiB.'); return JSON.parse(await file.text()); }
    $('report-file').onchange = e => { const file = e.target.files[0]; e.target.value = ''; if (file) action('Reading report…', async () => { await openReport(await fileJSON(file)); }); };
    $('inventory-file').onchange = e => { const file = e.target.files[0]; e.target.value = ''; if (file) action('Matching local discovery results…', async () => {
        if (!report) throw new Error('Open your billing report first.');
        const next = await fileJSON(file); Adoption.results(next, items); inventory = next; rebuild();
        const matched = items.filter(r => r.discovery); $('inventory-summary').textContent = `${matched.length} resource(s) matched; ${next.resources.length - matched.length} outside this report. ${matched.filter(r => r.discovery.status === 'verified').length} verified, ${matched.filter(r => r.discovery.status === 'imported').length} imported. Result supplied by your local tool; no cloud connection is active in this page.`;
    }); };
    $('apply-scope').onclick = () => action('Updating resource scope…', async () => { rebuild(); });
    $('resource-filter').onchange = renderRows;
    $('select-ready').onclick = () => { if (report?.sampleData) return; selected = new Set(items.filter(r => r.status === 'candidate' && !['imported', 'imported-needs-review', 'import-state-unknown', 'already-managed'].includes(r.discovery?.status)).slice(0, Adoption.MAX_BATCH).map(Adoption.identity)); renderRows(); };
    $('clear-selection').onclick = () => { selected.clear(); renderRows(); };
    $('download-package').onclick = () => action('Preparing your adoption package…', async () => {
        const manifest = Adoption.manifest(report, items.filter(r => selected.has(Adoption.identity(r))));
        const response = await fetch('adoption_tool.py', {cache: 'no-cache'}); if (!response.ok) throw new Error('The adoption tool could not be downloaded. Retry when the site is available.');
        const tool = await response.text(); if (!tool.includes('def discover(') || !tool.includes('def check_plan(')) throw new Error('The downloaded tool is invalid. Reload the site and retry.');
        const readme = '# Cloud Cost Atlas adoption package\n\n1. Install Python 3.10+, Terraform 1.5+ or OpenTofu, and the AWS/gcloud CLI.\n2. Log in to the cloud account/project listed in candidates.json. Discovery uses read-only API calls.\n3. Run: python adoption_tool.py discover --manifest candidates.json --output adoption\n   For a named AWS profile, add --aws-profile YOUR_PROFILE.\n4. Read adoption/README.md. Initialize Terraform, generate configuration, review the import-only plan, and adopt through the guarded tool.\n\nBilling resources may be stale, unsupported, or already managed. Use --managed-ids managed.json to exclude known managed resources (same resources identity format). Verify unmanaged ownership before importing. Configure a protected remote backend before shared use.\n\nReopen adoption/inventory.json on the Atlas adoption page to view progress. Never upload raw Terraform state/plans or secrets.\n';
        const url = URL.createObjectURL(Adoption.zip({'candidates.json': JSON.stringify(manifest, null, 2), 'adoption_tool.py': tool, 'README.md': readme})); const anchor = node('a'); anchor.href = url; anchor.download = 'atlas-adoption.zip'; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    action('Loading the accepted report from this browser…', async () => {
        $('report-summary').textContent = 'Import your billing CSVs or open a report JSON to discover adoption candidates. Older reports need to be regenerated to preserve account and region scope.';
        const saved = await ReportStore.get();
        if (saved) await openReport(saved);
    });
})();
