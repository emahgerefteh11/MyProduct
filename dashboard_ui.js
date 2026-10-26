/* Responsive workspace interactions and accessible charts, rendered locally. */
(() => {
    'use strict';
    const $ = id => document.getElementById(id);
    const node = (tag, value, className) => { const el = document.createElement(tag); if (value != null) el.textContent = value; if (className) el.className = className; return el; };
    const colors = ['#315e46', '#a7c692', '#d2bc86', '#75998f', '#c3d0bb'];
    const views = {
        overview: ['Spend overview', 'Understand where your cloud spend goes, and what comes next.'],
        products: ['Product costs', 'Follow reported spend from your platforms to the products they support.'],
        finops: ['Optimization', 'Explore opportunities with a clear view of the evidence behind them.'],
        compute: ['Compute & sizing', 'Understand the instance footprint represented in your billing report.'],
        naming: ['Resources & naming', 'Explore report-wide resource identifiers and naming conventions.'],
        iac: ['IaC Builder', 'Adopt existing infrastructure or prepare a starter template for a new workload.'],
    };
    const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
    const money = value => fmtCurrency(number(value));
    const compactMoney = value => {
        const currency = reportData?.analysis?.currency || (isDemoMode ? 'USD' : null);
        return new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1, ...(currency ? {style: 'currency', currency} : {})}).format(value);
    };
    function emptyChart(target, title, description) {
        const block = node('div', null, 'chart-empty'); block.append(node('strong', title), node('span', description)); target.replaceChildren(block);
    }
    function bars(target, rows, format, onSelect = null, selected = '') {
        target.replaceChildren();
        const max = Math.max(...rows.map(row => Math.abs(number(row.value))), 0), signed = rows.some(row => number(row.value) < 0);
        const list = node('div', null, 'rank-list');
        for (const row of rows) {
            const value = number(row.value), wrapper = node(onSelect ? 'button' : 'div', null, 'rank-row' + (selected === row.label ? ' selected' : ''));
            if (onSelect) { wrapper.type = 'button'; wrapper.setAttribute('aria-label', `Explore ${row.label}, ${format(value)}`); wrapper.onclick = () => onSelect(row.label); }
            const line = node('div', null, 'rank-line'); line.append(node('span', row.label, 'rank-label'), node('span', format(value), 'rank-value'));
            const track = node('div', null, 'bar-track' + (signed ? ' signed' : '')), fill = node('div', null, 'bar-fill' + (value < 0 ? ' negative' : ''));
            track.setAttribute('aria-hidden', 'true'); fill.style.width = `${max ? Math.abs(value) / max * (signed ? 50 : 100) : 0}%`; if (!value) fill.style.minWidth = '0';
            track.append(fill); wrapper.append(line, track); list.append(wrapper);
        }
        target.append(list);
        if (signed) target.append(node('p', 'Negative costs are shown to the left of the baseline. Costs may include credits or adjustments.', 'chart-footnote'));
    }

    window.renderAtlasCharts = (scope, data, tco) => {
        const target = $('chart-tco');
        const entries = Object.entries(tco || {}).map(([label, value]) => ({label, value: number(value)})).filter(r => r.value !== 0);
        const total = entries.reduce((n, r) => n + r.value, 0);
        if (!entries.length) emptyChart(target, 'No service costs to visualize', 'Costs will appear when service values are supplied for this scope.');
        else if (entries.some(r => r.value < 0)) bars(target, entries, money);
        else {
            target.replaceChildren();
            const layout = node('div', null, 'donut-layout'), donut = node('div', null, 'donut');
            let position = 0;
            const segments = entries.map((r, index) => { const start = position; position += r.value / total * 100; return `${colors[index % colors.length]} ${start}% ${position}%`; });
            donut.style.background = `conic-gradient(${segments.join(',')})`; donut.setAttribute('role', 'img'); donut.setAttribute('aria-label', 'Reported cost mix: ' + entries.map(r => `${r.label} ${money(r.value)}`).join(', '));
            const center = node('div', null, 'donut-center'); center.append(node('span', 'Total cost'), node('strong', compactMoney(total))); donut.append(center);
            const legend = node('div', null, 'chart-legend');
            entries.forEach((r, i) => { const item = node('div', null, 'legend-item'), dot = node('span', null, 'dot'); dot.style.background = colors[i % colors.length]; dot.setAttribute('aria-hidden', 'true'); const label = node('div', r.label); label.append(node('small', (r.value / total * 100).toFixed(1) + '%')); item.append(dot, label, node('strong', money(r.value))); legend.append(item); });
            layout.append(donut, legend); target.append(layout);
        }
        const platforms = (data.platforms || []).map(p => ({label: String(p.Platform), value: p.BilledCost})).sort((a, b) => b.value - a.value);
        const selected = scope === 'all' ? '' : scope;
        if (!platforms.length) emptyChart($('chart-platform'), 'No platform allocation yet', 'Map a platform or team column when importing billing.');
        else {
            // Keep the selected platform visible even when outside the top eight.
            const top = platforms.slice(0, 8), current = platforms.find(p => p.label === selected);
            if (current && !top.includes(current)) top[top.length - 1] = current;
            bars($('chart-platform'), top, money, label => {
                $('global-platform-filter').value = label; applyFilters(); $('global-platform-filter').focus();
            }, selected);
            if (platforms.length > 8) $('chart-platform').append(node('p', `Showing eight of ${platforms.length} platforms, including your selection. Use the platform filter for the full list.`, 'chart-footnote'));
        }
        if (data.analysis?.importMethod === 'browser') emptyChart($('chart-skus'), 'Sizing data is not included', 'Browser imports summarize billing costs. Use the command-line analyzer for supported SKU and run-hour details.');
        else {
            const skuMap = new Map();
            for (const p of data.platforms || []) if (scope === 'all' || p.Platform === scope) for (const sku of p.vmSkus || []) skuMap.set(String(sku.sku), (skuMap.get(String(sku.sku)) || 0) + number(sku.hours));
            const skus = [...skuMap].map(([label, value]) => ({label, value})).sort((a, b) => b.value - a.value).slice(0, 10);
            if (!skus.length) emptyChart($('chart-skus'), 'No supported compute SKUs', 'This scope does not contain supported instance run-hour records.');
            else bars($('chart-skus'), skus, n => new Intl.NumberFormat('en-US', {maximumFractionDigits: 1}).format(n) + ' hrs');
        }
    };

    const sidebar = $('workspace-sidebar'), main = $('main-content'), media = window.matchMedia('(max-width: 1040px)');
    let navigationOpen = false;
    function setNavigation(open, restore = false) {
        navigationOpen = !!open && media.matches;
        document.body.classList.toggle('nav-open', navigationOpen);
        $('nav-toggle').setAttribute('aria-expanded', String(navigationOpen));
        $('nav-backdrop').hidden = !navigationOpen;
        sidebar.inert = media.matches && !navigationOpen;
        main.inert = navigationOpen;
        if (navigationOpen) { sidebar.setAttribute('role', 'dialog'); sidebar.setAttribute('aria-modal', 'true'); $('nav-close').focus(); }
        else { sidebar.removeAttribute('role'); sidebar.removeAttribute('aria-modal'); if (restore) $('nav-toggle').focus(); }
    }
    $('nav-toggle').onclick = () => setNavigation(!navigationOpen, true);
    $('nav-close').onclick = () => setNavigation(false, true);
    $('nav-backdrop').onclick = () => setNavigation(false, true);
    media.addEventListener('change', () => setNavigation(false));
    document.addEventListener('keydown', event => {
        if (!navigationOpen) return;
        if (event.key === 'Escape') { event.preventDefault(); setNavigation(false, true); }
        if (event.key === 'Tab') {
            const controls = [...sidebar.querySelectorAll('a[href],button:not([disabled])')].filter(el => el.getClientRects().length);
            const first = controls[0], last = controls.at(-1);
            if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
    });
    setNavigation(false);

    window.syncAtlasView = view => {
        if (!views[view] || !$('page-description')) return;
        $('view-breadcrumb').textContent = views[view][0]; $('page-description').textContent = views[view][1];
        document.title = `${views[view][0]} · Cloud Cost Atlas`;
        document.querySelector('.filter-bar').hidden = ['iac', 'naming'].includes(view);
    };
    function activate(view, focus = true, history = true) {
        if (!views[view] || !reportData || !$(`tab-${view}`)) return;
        switchTab(view, document.querySelector(`[data-view="${view}"]`));
        setNavigation(false);
        if (history && window.location.hash !== '#' + view) window.history.pushState(null, '', '#' + view);
        if (focus) $('page-title').focus({preventScroll: true});
    }
    document.querySelectorAll('[data-view]').forEach(link => link.addEventListener('click', event => { event.preventDefault(); activate(link.dataset.view); }));
    window.addEventListener('hashchange', () => { if (window.location.hash === '#connections') $('cloud-dialog').showModal(); else activate(window.location.hash.slice(1) || 'overview', true, false); });
    document.addEventListener('click', event => {
        const close = event.target.closest('[data-close-dialog]');
        if (close) close.closest('dialog').close();
        const cloud = event.target.closest('[data-open-cloud]');
        if (cloud) { setNavigation(false, navigationOpen); $('cloud-dialog').showModal(); }
        const product = event.target.closest('[data-product-index]');
        if (product) openProduct(Number(product.dataset.productIndex));
        const investigate = event.target.closest('[data-investigate-index]');
        if (investigate) {
            const recommendation = window.finopsRecommendations[Number(investigate.dataset.investigateIndex)];
            if (recommendation) { investigateAnomaly(recommendation.product); window.history.pushState(null, '', '#iac'); $('page-title').focus(); }
        }
        document.querySelectorAll('.report-actions[open]').forEach(details => { if (!details.contains(event.target)) details.open = false; });
    });
    for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } });
    if (window.location.hash === '#connections') $('cloud-dialog').showModal();

    function openProduct(index) {
        const p = window.currentProductRows?.[index]; if (!p) return;
        $('product-dialog-title').textContent = p.Product;
        $('product-dialog-platform').textContent = `Platform: ${p.Platform || 'Not specified'}`;
        const environments = (p.environments || []).map(e => e.Environment).join(', ') || 'Not supplied';
        const providers = (p.providers || []).map(r => String(r.ProviderName).toUpperCase()).join(', ') || 'Not supplied';
        const metrics = [['Reported spend', money(p.BilledCost)], ['Environment', environments], ['Cloud provider', providers]];
        $('product-dialog-metrics').replaceChildren(...metrics.map(([label, value]) => { const el = node('div'); el.append(node('span', label), node('strong', value)); return el; }));
        const services = (p.services || []).map(s => ({label: String(s.ServiceCategory), value: s.BilledCost})).sort((a, b) => b.value - a.value).slice(0, 10);
        if (services.length) bars($('product-dialog-services'), services, money); else emptyChart($('product-dialog-services'), 'No service breakdown was supplied', 'This report includes a product total without service attribution.');
        $('product-dialog-evidence').textContent = isDemoMode ? 'Sample report. Any optimization examples elsewhere in the demo are simulated.' : 'Reported costs and explicit allocation. Billing alone does not establish utilization or optimization savings.';
        $('product-dialog').showModal();
    }
    window.refreshAtlas = () => {
        if (!reportData || !$('scope-summary')) return;
        main.setAttribute('aria-busy', 'false');
        const scope = $('global-platform-filter').value || 'all', analysis = reportData.analysis;
        let timing = 'Analysis time unavailable';
        if (isDemoMode) timing = 'Sample data · No live account connection';
        else if (analysis?.generatedAt && Number.isFinite(Date.parse(analysis.generatedAt))) timing = 'Analyzed ' + new Date(analysis.generatedAt).toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'});
        const period = analysis?.usagePeriod;
        if (!isDemoMode && period?.start && Number.isFinite(Date.parse(period.start))) timing += ' · Usage from ' + new Date(period.start).toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'}) + (period.complete ? '' : ' (incomplete)');
        const dot = node('span', null, 'dot'); dot.setAttribute('aria-hidden', 'true');
        $('scope-summary').replaceChildren(dot, node('span', `${scope === 'all' ? 'All platforms' : scope} · ${timing}`));
        const warningCount = analysis?.warnings?.length || 0;
        const summary = document.querySelector('.quality-panel summary'); summary.textContent = 'Report provenance & data quality' + (warningCount ? ` · ${warningCount} note${warningCount === 1 ? '' : 's'}` : '');
    };
    // Embedded demos can finish rendering before this deferred script executes.
    if (typeof reportData !== 'undefined' && reportData) {
        window.refreshAtlas(); renderCharts($('global-platform-filter').value || 'all');
        activate(window.location.hash.slice(1) || 'overview', false, false);
    } else {
        // init's fetch is asynchronous. Apply the initial deep link once the
        // report becomes visible, without moving focus away from user input.
        const observer = new MutationObserver(() => {
            if (typeof reportData !== 'undefined' && reportData && $('report-content') && !$('report-content').hidden) {
                activate(window.location.hash.slice(1) || 'overview', false, false); observer.disconnect();
            } else if (!$('report-content')) observer.disconnect();
        });
        observer.observe(main, {childList: true, subtree: true, attributes: true, attributeFilter: ['hidden']});
    }
})();
