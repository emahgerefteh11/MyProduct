/* Shared formatting for the difference between source rows and the rounded report. */
(function (root) {
    'use strict';
    function roundingDifference(reconciliation, currency) {
        const exact = reconciliation?.exactDifference;
        const hasExact = typeof exact === 'string' && exact.length <= 512 && /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(exact);
        const numeric = reconciliation?.difference;
        if (!hasExact && !Number.isFinite(numeric)) return 'unavailable';
        const unit = typeof currency === 'string' && /^[A-Z]{3}$/.test(currency) ? ` ${currency}` : '';
        const negative = hasExact ? exact.startsWith('-') : numeric < 0;
        const magnitude = hasExact ? exact.replace(/^-/, '') : null;
        if (hasExact ? !/[1-9]/.test(magnitude) : numeric === 0) return 'none';
        const approximate = hasExact ? Number(magnitude) : Math.abs(numeric);
        if (!Number.isFinite(approximate)) return 'unavailable';
        const sign = negative ? '−' : '';
        if (approximate < 0.00000001) return `less than 0.00000001${unit} (report is ${negative ? 'lower' : 'higher'})`;
        if (hasExact && magnitude.length <= 20) return `${sign}${magnitude}${unit}`;
        const shown = new Intl.NumberFormat('en-US', {useGrouping: false, maximumFractionDigits: 8}).format(approximate);
        return `${hasExact ? 'about ' : ''}${sign}${shown}${unit}`;
    }
    root.BillingFormat = {roundingDifference};
})(globalThis);
