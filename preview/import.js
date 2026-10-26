'use strict';
(() => {
    const $ = id => document.getElementById(id);
    let selected = [], prepared = [], configs = [], validation = null, worker = null, sequence = 0, busy = false;
    const pending = new Map();
    function node(tag, text, className) { const el = document.createElement(tag); if (text != null) el.textContent = text; if (className) el.className = className; return el; }
    function showError(message) { $('error').textContent = message; $('error').hidden = false; $('error').focus(); }
    function resetError() { $('error').hidden = true; $('error').textContent = ''; }
    function canAccept() { return validation?.ok && (!validation.warnings.length || $('acknowledge').checked); }
    function buttons() {
        document.querySelectorAll('button, select, input').forEach(el => { el.disabled = busy; });
        $('read-files').disabled = busy || !selected.length;
        $('download-report').disabled = busy || !canAccept();
        $('open-report').disabled = busy || !canAccept();
    }
    function showStep(step) {
        for (let n = 1; n <= 3; n++) {
            $('step-' + n).hidden = n !== step;
            if (n === step) $('step-label-' + n).setAttribute('aria-current', 'step');
            else $('step-label-' + n).removeAttribute('aria-current');
        }
        $({1: 'choose-title', 2: 'mapping-title', 3: 'preview-title'}[step]).focus();
    }
    function ask(type, payload) {
        if (!worker) {
            worker = new Worker('billing_worker.js');
            worker.onmessage = ({data}) => {
                const promise = pending.get(data.id); if (!promise) return;
                pending.delete(data.id); data.error ? promise.reject(new Error(data.error)) : promise.resolve(data.result);
            };
            worker.onerror = () => {
                for (const promise of pending.values()) promise.reject(new Error('The import worker could not run. Reload this page and try smaller files.'));
                pending.clear(); worker.terminate(); worker = null;
            };
        }
        return new Promise((resolve, reject) => { const id = ++sequence; pending.set(id, {resolve, reject}); worker.postMessage({id, type, ...payload}); });
    }
    async function action(message, task) {
        if (busy) return;
        busy = true; resetError(); $('status').textContent = message; $('status').hidden = false; buttons();
        try { await task(); } catch (e) { showError(e.message); }
        finally { busy = false; $('status').hidden = true; buttons(); }
    }
    function choose(files) {
        if (busy) return;
        resetError();
        const next = [...selected, ...Array.from(files)];
        if (next.length > BillingImport.LIMITS.files) { showError('Choose at most 10 CSV files. Remove a file before adding more.'); return; }
        if (next.reduce((n, f) => n + f.size, 0) > BillingImport.LIMITS.bytes) { showError('Choose at most 20 MiB of files combined. Use smaller exports or the command-line analyzer.'); return; }
        if (next.some(f => !/\.csv$/i.test(f.name))) { showError('Choose uncompressed .csv files. Unzip compressed exports first.'); return; }
        selected = next; validation = null; renderFiles();
    }
    function renderFiles() {
        $('selected-files').replaceChildren();
        selected.forEach((file, index) => {
            const li = node('li'); li.append(node('span', `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`));
            const remove = node('button', 'Remove', 'secondary'); remove.type = 'button'; remove.setAttribute('aria-label', `Remove ${file.name}`);
            remove.onclick = () => { selected.splice(index, 1); validation = null; renderFiles(); };
            li.append(remove); $('selected-files').append(li);
        }); buttons();
    }
    function selectField(parent, label, id, choices, value, onChange, required = false) {
        const wrapper = node('label', label, 'field' + (required ? ' required' : '')); wrapper.htmlFor = id;
        const select = node('select'); select.id = id;
        for (const [val, title] of choices) { const option = node('option', title); option.value = val; select.append(option); }
        select.value = value; select.onchange = () => { validation = null; onChange(select.value); buttons(); };
        wrapper.append(select); parent.append(wrapper); return select;
    }
    function renderMappings() {
        $('mappings').replaceChildren();
        prepared.forEach((file, i) => {
            const config = configs[i], section = node('section', null, 'mapping-file');
            section.append(node('h3', file.name), node('p', `${file.rows.toLocaleString()} rows · ${file.headers.length} columns · Detected: ${BillingImport.FORMATS[file.detected.format]}`, 'muted small'));
            const basic = node('div', null, 'mapping-grid');
            selectField(basic, 'Export format', `format-${i}`, Object.entries(BillingImport.FORMATS), config.format, value => { configs[i] = BillingImport.detect(file.headers, value); renderMappings(); });
            selectField(basic, 'Provider when no provider value is present', `provider-${i}`, [['unknown', 'Unknown / not specified'], ['aws', 'AWS'], ['azure', 'Azure'], ['gcp', 'Google Cloud']], config.provider, value => { config.provider = value; });
            selectField(basic, 'Cost basis', `basis-${i}`, [['billed', 'Billed cost'], ['effective', 'Effective cost']], config.costBasis, value => { config.costBasis = value; });
            section.append(basic);
            const grid = node('div', null, 'mapping-grid');
            const advanced = node('details', null, 'optional-mappings'); advanced.append(node('summary', 'Resource IDs, dates, and tags (optional)'));
            const optional = node('div', null, 'mapping-grid');
            for (const [key, label] of Object.entries(BillingImport.FIELDS)) {
                selectField(['cost', 'currency', 'service', 'product', 'platform', 'environment'].includes(key) ? grid : optional, label, `map-${i}-${key}`, [['', key === 'cost' ? 'Choose a cost column…' : 'Not mapped'], ...file.headers.map(h => [h, h])], config.mapping[key], value => {
                    config.mapping[key] = value;
                    if (key === 'cost' && /^(billedcost|effectivecost)$/i.test(value)) { config.costBasis = value.toLowerCase() === 'effectivecost' ? 'effective' : 'billed'; $(`basis-${i}`).value = config.costBasis; }
                }, key === 'cost');
            }
            advanced.append(optional); section.append(grid, advanced);
            const details = node('details'); details.append(node('summary', 'Preview the first 5 rows'));
            const scroll = node('div', null, 'table-scroll'), table = node('table', null, 'sample'), head = node('thead'), header = node('tr'), body = node('tbody');
            file.headers.forEach(h => header.append(node('th', h))); head.append(header);
            file.sample.forEach(row => { const tr = node('tr'); row.cells.forEach(c => { const td = node('td', c.length > 180 ? c.slice(0, 180) + '…' : c); tr.append(td); }); body.append(tr); });
            table.append(head, body); scroll.append(table); details.append(scroll); section.append(details); $('mappings').append(section);
        });
    }
    function tableRows(id, rows) { $(id).replaceChildren(); rows.forEach(values => { const tr = node('tr'); values.forEach(value => tr.append(node('td', String(value)))); $(id).append(tr); }); }
    function renderPreview() {
        const result = validation, report = result.report;
        $('acknowledge').checked = false;
        $('validation-badge').textContent = result.ok ? 'READY TO REVIEW' : 'NEEDS CORRECTION';
        $('validation-badge').className = 'badge' + (result.ok ? '' : ' invalid');
        $('validation-summary').textContent = result.ok ? `${result.rowCount.toLocaleString()} billing rows validated. Review totals and warnings before opening your report.` : `${result.errorCount.toLocaleString()} issue(s) found. Your existing report has not changed. Edit mappings or correct the CSV and try again.`;
        $('validation-errors').hidden = result.ok; $('valid-preview').hidden = !result.ok; $('storage-note').hidden = !result.ok;
        tableRows('error-rows', result.errors.map(e => [e.file || 'All files', e.line || '—', e.column || '—', e.message]));
        $('error-limit').textContent = result.errorCount > result.errors.length ? `Showing the first ${result.errors.length} issues.` : '';
        $('warning-list').replaceChildren(...result.warnings.map(w => node('li', w)));
        $('warning-panel').hidden = !result.warnings.length; $('acknowledge-label').hidden = !result.ok || !result.warnings.length;
        if (report) {
            const fmt = amount => new Intl.NumberFormat(undefined, report.analysis.currency ? {style: 'currency', currency: report.analysis.currency} : {minimumFractionDigits: 2, maximumFractionDigits: 2}).format(amount);
            $('preview-metrics').replaceChildren();
            for (const [label, value] of [[`Total ${report.analysis.costBasis} cost`, fmt(report.TotalBilledCost)], ['Billing rows', report.analysis.rowCount.toLocaleString()], ['Currency', report.analysis.currency || 'Unspecified']]) {
                const card = node('div', null, 'metric'); card.append(node('span', label), node('strong', value)); $('preview-metrics').append(card);
            }
            const period = report.analysis.usagePeriod;
            $('preview-period').textContent = `Observed usage: ${period.start || 'Unknown start'} → ${period.end || 'Unknown end'}${period.complete ? '' : ' (incomplete)'}.`;
            tableRows('source-rows', report.analysis.sources.map(s => [s.file, BillingImport.FORMATS[s.schema], s.rows.toLocaleString(), fmt(Number(s.sourceCostTotal)), s.costBasis]));
            const r = report.analysis.reconciliation;
            $('reconciliation').textContent = `Source sum: ${fmt(r.sourceTotal)} · Report sum: ${fmt(r.reportTotal)} · Rounding difference: ${fmt(r.difference)}. This checks imported amounts, not completeness against your cloud invoice.`;
            tableRows('service-rows', report.services.slice(0, 8).map(s => [s.ServiceCategory, fmt(s.BilledCost)]));
            tableRows('allocation-rows', [['Product', fmt(report.allocation.unallocatedProductCost)], ['Platform / team', fmt(report.allocation.unallocatedPlatformCost)], ['Environment', fmt(report.allocation.unallocatedEnvironmentCost)]]);
        } buttons();
    }
    $('billing-files').onchange = e => { choose(e.target.files); e.target.value = ''; };
    const drop = $('drop-zone');
    drop.ondragover = e => { e.preventDefault(); if (!busy) drop.classList.add('dragging'); };
    drop.ondragleave = () => drop.classList.remove('dragging');
    drop.ondrop = e => { e.preventDefault(); drop.classList.remove('dragging'); choose(e.dataTransfer.files); };
    $('read-files').onclick = () => action('Reading files and detecting columns…', async () => {
        prepared = await ask('prepare', {files: selected, delimiter: $('delimiter').value === 'tab' ? '\t' : $('delimiter').value});
        configs = prepared.map(f => structuredClone(f.detected)); validation = null; renderMappings(); showStep(2);
    });
    $('validate').onclick = () => action('Validating every billing row and calculating your preview…', async () => {
        validation = await ask('validate', {configs, currency: $('currency-assumption').value}); renderPreview(); showStep(3);
    });
    $('currency-assumption').oninput = () => { validation = null; buttons(); };
    $('back-files').onclick = () => { resetError(); validation = null; showStep(1); buttons(); };
    $('back-mappings').onclick = () => { resetError(); validation = null; showStep(2); buttons(); };
    $('acknowledge').onchange = buttons;
    $('download-report').onclick = () => {
        if (!canAccept()) return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(validation.report, null, 2)], {type: 'application/json'}));
        const anchor = node('a'); anchor.href = url; anchor.download = 'report_data.json'; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    $('open-report').onclick = () => action('Saving your accepted report in this browser…', async () => {
        if (!canAccept()) return;
        await ReportStore.save(validation.report); location.href = 'dashboard.html?source=import';
    });
    if (!window.Worker || !window.crypto?.subtle) showError('Use a current browser over HTTPS (or localhost) to import billing files.');
    buttons();
})();
