/* Design-preview presentation. Billing values remain in the shared engines. */
(() => {
    'use strict';
    const $ = id => document.getElementById(id);
    if ($('return-report') && typeof ReportStore !== 'undefined') {
        ReportStore.get().then(report => { $('return-report').hidden = !report; }).catch(() => {});
    }
    if ($('spend-hero')) {
        let currentView = document.querySelector('[data-view].active')?.dataset.view || 'overview';
        const sync = window.syncAtlasView;
        window.syncAtlasView = view => {
            sync?.(view);
            currentView = view;
            $('spend-hero').hidden = view !== 'overview';
            if (view === 'iac') {
                $('page-title').textContent = 'Manage as code';
                $('view-breadcrumb').textContent = 'Manage as code';
                document.title = 'Manage as code · Cloud Cost Atlas';
            }
            window.refreshPreview();
        };
        window.refreshPreview = () => {
            if (typeof reportData === 'undefined' || !reportData || !$('spend-hero')) return;
            $('spend-hero').hidden = currentView !== 'overview';
            const scope = $('global-platform-filter').value || 'all';
            const platform = (reportData.platforms || []).find(p => p.Platform === scope);
            const total = Number(scope === 'all' ? reportData.TotalBilledCost : platform?.BilledCost) || 0;
            const analysis = reportData.analysis, period = analysis?.usagePeriod;
            const date = value => value && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'}) : null;
            const start = date(period?.start), end = date(period?.end);
            const range = start && end ? `${start} – ${end}${period.complete ? '' : ' · Incomplete period'}` : 'Billing period not supplied';
            const currency = analysis?.currency || (isDemoMode ? 'USD illustrated' : 'Currency not supplied');
            $('preview-period').textContent = `${range} · ${currency}`;
            $('kpi-total-label').textContent = analysis?.costBasis === 'effective' ? 'Effective spend' : 'Reported spend';
            const entries = Object.entries(window.currentTco || {}).map(([label, value]) => [label, Number(value)]).filter(([, value]) => Number.isFinite(value));
            const largest = entries.filter(([, value]) => value > 0).sort((a, b) => b[1] - a[1])[0];
            const hasCredits = entries.some(([, value]) => value < 0);
            if (largest) {
                const [label, amount] = largest;
                const share = total > 0 && !hasCredits && amount <= total ? (amount / total * 100).toFixed(1) : null;
                $('preview-driver').textContent = share ? `${label} accounts for ${share}%` : `${label} is the largest charge`;
                $('preview-driver-detail').textContent = `${fmtCurrency(amount)} in ${label.toLowerCase()} costs${scope === 'all' ? ' across your report' : ' in ' + scope}.${hasCredits ? ' Net spend includes credits or adjustments.' : ''}`;
            } else {
                $('preview-driver').textContent = total < 0 ? 'Net credits in this scope' : 'No positive service costs';
                $('preview-driver-detail').textContent = 'Review the service breakdown for the amounts supplied in your report.';
            }
            const source = isDemoMode ? 'Sample report · No account connected' : analysis?.importMethod === 'browser' ? 'Imported on this device' : 'Site report';
            $('scope-summary').textContent = source + (currentView === 'overview' ? '' : ` · ${range} · ${currency}`);
            const warnings = analysis?.warnings || [];
            const summary = document.querySelector('.quality-panel summary');
            summary.textContent = 'Report details' + (warnings.length ? ` · ${warnings.length} data note${warnings.length === 1 ? '' : 's'}` : '');
            $('preview-provider-detail').textContent = 'Reported clouds (whole report): ' + ((reportData.providers || []).map(p => `${String(p.ProviderName).toUpperCase()} ${fmtCurrency(p.BilledCost)}`).join(' · ') || 'Not supplied');
            const alert = $('preview-alert');
            alert.replaceChildren();
            const notices = warnings.slice(0, 2).map(String);
            if (!isDemoMode && !analysis?.currency) notices.unshift('Billing currency was not supplied. Amounts are shown without currency conversion.');
            if (!isDemoMode && period && !period.complete) notices.push('Usage dates are incomplete; totals may cover part of a billing period.');
            alert.hidden = !notices.length;
            if (notices.length) {
                alert.append(document.createTextNode(notices.join(' ') + ' '));
                const link = document.createElement('a'); link.href = '#report-details'; link.textContent = 'Review report details';
                link.onclick = event => { event.preventDefault(); const details = document.querySelector('.quality-panel details'); details.open = true; details.querySelector('summary').focus(); details.scrollIntoView({block: 'start'}); };
                alert.append(link);
            }
        };
        document.querySelector('.quality-panel').id = 'report-details';
        window.refreshPreview();
        // A saved report can finish loading after deferred UI scripts execute.
        const observer = new MutationObserver(() => {
            if (!$('report-content').hidden) { window.refreshPreview(); observer.disconnect(); }
        });
        if ($('report-content').hidden) observer.observe($('report-content'), {attributes: true, attributeFilter: ['hidden']});
    }

    // Instructions become visible after a package has actually been prepared.
    if ($('local-steps')) {
        document.querySelectorAll('#local-steps pre').forEach((pre, index) => {
            const block = document.createElement('div'); block.className = 'command-block';
            const bar = document.createElement('div'); bar.className = 'command-bar';
            const label = document.createElement('span'); label.textContent = index === 0 ? 'Discover and prepare a plan' : 'Import after reviewing the plan';
            const button = document.createElement('button'); button.type = 'button'; button.className = 'command-copy'; button.textContent = 'Copy commands'; button.setAttribute('aria-label', 'Copy ' + label.textContent.toLowerCase());
            const status = document.createElement('p'); status.className = 'copy-status'; status.setAttribute('role', 'status');
            button.onclick = async () => {
                try { await navigator.clipboard.writeText(pre.textContent); status.textContent = 'Commands copied.'; }
                catch { status.textContent = 'Select the commands below and copy them with your keyboard.'; }
            };
            pre.before(block); bar.append(label, button); block.append(bar, status, pre);
        });
    }
})();
