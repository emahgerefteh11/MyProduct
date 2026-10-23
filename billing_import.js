/* Browser billing ingestion. No network access; also exercised directly in Node. */
(function (root) {
    'use strict';
    const LIMITS = {files: 10, bytes: 20 * 1024 * 1024, rows: 100000, columns: 500, errors: 100};
    const FIELDS = {
        cost: 'Cost amount (required)', currency: 'Billing currency', service: 'Service',
        provider: 'Cloud provider', resource: 'Resource ID or name', product: 'Product',
        platform: 'Platform / team', environment: 'Environment', tags: 'Tags / labels (JSON)',
        account: 'Resource account / project ID', region: 'Resource region', zone: 'Resource availability zone',
        usage_start: 'Usage start', usage_end: 'Usage end', billing_start: 'Billing period start', billing_end: 'Billing period end',
    };
    const FORMATS = {aws: 'AWS CUR', azure: 'Azure Cost Management', gcp: 'Google Cloud billing', focus: 'FOCUS', custom: 'Custom CSV'};

    function parseCSV(text, delimiter = ',') {
        if (![',', ';', '\t'].includes(delimiter)) throw new Error('Choose comma, semicolon, or tab as the delimiter.');
        text = text.replace(/^\uFEFF/, '');
        const records = [];
        let cells = [], value = '', quoted = false, closed = false, line = 1, start = 1;
        function field() { cells.push(value); value = ''; closed = false; }
        function record() {
            field();
            if (cells.some(cell => cell.trim() !== '')) records.push({cells, line: start});
            cells = [];
            if (records.length > LIMITS.rows + 1) throw new Error(`Use at most ${LIMITS.rows.toLocaleString()} rows per import. For larger exports, use the command-line analyzer.`);
        }
        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            if (quoted) {
                if (ch === '"') {
                    if (text[i + 1] === '"') { value += '"'; i++; } else { quoted = false; closed = true; }
                } else { value += ch; if (ch === '\n') line++; }
                continue;
            }
            if (closed && ch !== delimiter && ch !== '\r' && ch !== '\n') throw new Error(`Line ${line}: unexpected text after a closing quote.`);
            if (ch === '"') {
                if (value) throw new Error(`Line ${line}: quotes must surround the entire field.`);
                quoted = true;
            } else if (ch === delimiter) field();
            else if (ch === '\n' || ch === '\r') {
                record();
                if (ch === '\r' && text[i + 1] === '\n') i++;
                line++; start = line;
            } else value += ch;
        }
        if (quoted) throw new Error(`Line ${start}: an opening quote has no closing quote.`);
        if (value || cells.length || closed) record();
        if (!records.length) throw new Error('This file is empty. Export a CSV with a header and billing rows.');
        const headers = records.shift().cells.map(h => h.trim());
        if (headers.length > LIMITS.columns) throw new Error(`At most ${LIMITS.columns} columns are supported.`);
        if (headers.some(h => !h)) throw new Error('Every column needs a non-empty header.');
        if (new Set(headers.map(h => h.toLowerCase())).size !== headers.length) throw new Error('Column headers must be unique (ignoring case).');
        if (!records.length) throw new Error('This file has a header but no billing rows.');
        return {headers, records};
    }

    function detect(headers, format) {
        const pick = (...names) => names.map(n => headers.find(h => h.toLowerCase() === n.toLowerCase())).find(Boolean) || '';
        const inferred = pick('BilledCost', 'EffectiveCost') ? 'focus' : pick('lineItem/UnblendedCost') ? 'aws'
            : pick('service.description', 'ServiceDescription', 'SkuDescription') ? 'gcp'
            : pick('CostInBillingCurrency', 'PreTaxCost', 'ConsumedService', 'MeterCategory', 'InstanceId') || (pick('Cost') && pick('ResourceId')) ? 'azure' : 'custom';
        format = format || inferred;
        const mapping = Object.fromEntries(Object.keys(FIELDS).map(k => [k, '']));
        Object.assign(mapping, {
            product: pick('resourceTags/user:Product', 'tags/Product', 'Product'),
            platform: pick('resourceTags/user:Platform', 'tags/Platform', 'Platform', 'Team'),
            environment: pick('resourceTags/user:Environment', 'tags/Environment', 'Environment'),
            tags: pick('Tags', 'Labels', 'project.labels'), provider: pick('ProviderName', 'Provider'),
            currency: pick('BillingCurrency', 'BillingCurrencyCode', 'Currency', 'lineItem/CurrencyCode'),
            resource: pick('ResourceId', 'ResourceName', 'InstanceId', 'lineItem/ResourceId', 'resource.global_name', 'resource.name', 'LineItem'),
            account: pick('lineItem/UsageAccountId', 'project.id', 'ProjectId', 'SubAccountId', 'SubscriptionId'),
            region: pick('product/regionCode', 'location.region', 'RegionId', 'Region'),
            zone: pick('lineItem/AvailabilityZone', 'location.zone', 'Zone'),
            usage_start: pick('ChargePeriodStart', 'lineItem/UsageStartDate', 'UsageStartDate', 'usage_start_time', 'Date', 'UsageDateTime'),
            usage_end: pick('ChargePeriodEnd', 'lineItem/UsageEndDate', 'UsageEndDate', 'usage_end_time'),
            billing_start: pick('BillingPeriodStart', 'BillingPeriodStartDate', 'bill/BillingPeriodStartDate'),
            billing_end: pick('BillingPeriodEnd', 'BillingPeriodEndDate', 'bill/BillingPeriodEndDate'),
        });
        const costNames = {aws: ['lineItem/UnblendedCost'], azure: ['CostInBillingCurrency', 'Cost', 'PreTaxCost'], gcp: ['Cost'], focus: ['BilledCost', 'EffectiveCost'], custom: ['Amount', 'Cost', 'BilledCost', 'EffectiveCost']};
        const serviceNames = {aws: ['lineItem/ProductCode', 'product/ProductName'], azure: ['ServiceName', 'ConsumedService', 'MeterCategory'], gcp: ['ServiceDescription', 'service.description', 'SkuDescription'], focus: ['ServiceName', 'ServiceCategory'], custom: ['Service', 'ServiceName', 'ServiceCategory']};
        mapping.cost = pick(...costNames[format]);
        mapping.service = pick(...serviceNames[format]);
        return {format, provider: ['aws', 'azure', 'gcp'].includes(format) ? format : 'unknown', costBasis: mapping.cost.toLowerCase() === 'effectivecost' ? 'effective' : 'billed', mapping};
    }

    // Exact decimal sums avoid accumulating binary floating-point rounding errors.
    function decimal(raw) {
        const text = String(raw).trim();
        if (text.length > 80 || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text) || !Number.isFinite(Number(text))) throw new Error('Enter a finite numeric amount, without currency symbols or thousands separators.');
        const [mantissa, exp = '0'] = text.toLowerCase().split('e');
        if (Math.abs(Number(exp)) > 324) throw new Error('The amount exponent is outside the supported range.');
        const fraction = (mantissa.split('.')[1] || '').length;
        let n = BigInt(mantissa.replace('.', '')), s = fraction - Number(exp);
        if (s < 0) { n *= 10n ** BigInt(-s); s = 0; }
        return {n, s};
    }
    const zero = () => ({n: 0n, s: 0});
    function add(a, b) {
        const s = Math.max(a.s, b.s);
        a.n = a.n * 10n ** BigInt(s - a.s) + b.n * 10n ** BigInt(s - b.s); a.s = s;
        return a;
    }
    function decimalText(a) {
        const sign = a.n < 0n ? '-' : '', digits = (a.n < 0n ? -a.n : a.n).toString().padStart(a.s + 1, '0');
        return sign + (a.s ? digits.slice(0, -a.s) + '.' + digits.slice(-a.s) : digits);
    }
    function amount(a) {
        const n = Number(decimalText(a));
        if (!Number.isFinite(n) || Math.abs(n) > Number.MAX_SAFE_INTEGER / 100) throw new Error('The cost total is too large to display accurately. Split this export into smaller reports.');
        return n;
    }
    function rounded(a) {
        if (a.s <= 2) return amount(a);
        const divisor = 10n ** BigInt(a.s - 2), sign = a.n < 0n ? -1n : 1n, absolute = a.n * sign;
        return amount({n: sign * ((absolute + divisor / 2n) / divisor), s: 2});
    }
    function providerName(value) {
        const p = String(value || '').trim().toLowerCase();
        if (/^(aws|amazon|amazon web services)(,? inc\.?)?$/.test(p)) return 'aws';
        if (['azure', 'microsoft', 'microsoft azure'].includes(p)) return 'azure';
        if (['gcp', 'google', 'google cloud', 'google cloud platform'].includes(p)) return 'gcp';
        return 'unknown';
    }
    function clean(value) {
        const text = String(value ?? '').trim();
        return ['', 'unknown', 'untagged', 'unallocated', 'null', 'none', 'n/a', 'nan'].includes(text.toLowerCase()) ? '' : text;
    }
    function tagsOf(value) {
        if (!value.trim()) return {};
        const data = JSON.parse(value.replace(/^project:labels:/, ''));
        const entries = Array.isArray(data) ? data.map(x => [x.key, x.value]) : data && typeof data === 'object' ? Object.entries(data) : null;
        if (!entries || entries.some(([k, v]) => typeof k !== 'string' || !['string', 'number', 'boolean'].includes(typeof v))) throw new Error('Tags must be a JSON object or an array of key/value objects.');
        return Object.fromEntries(entries.map(([k, v]) => [k.toLowerCase(), clean(v)]));
    }
    function dateOf(raw) {
        if (!raw) return null;
        let text = raw.trim();
        if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(text)) {
            const [m, d, y] = text.split('/'); text = `${y}-${m.padStart(2, '0')}-${d.padStart(2, '0')}`;
        }
        if (!/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/.test(text)) return null;
        const y = +text.slice(0, 4), m = +text.slice(5, 7), d = +text.slice(8, 10);
        if (m < 1 || m > 12 || d < 1 || d > new Date(Date.UTC(y, m, 0)).getUTCDate()) return null;
        if (text.length === 10) text += 'T00:00:00Z';
        else { text = text.replace(' ', 'T'); if (!/(Z|[+-]\d{2}:?\d{2})$/.test(text)) text += 'Z'; }
        return Number.isFinite(Date.parse(text)) ? new Date(text).toISOString() : null;
    }
    const SERVICE_RULES = [
        ['Block Storage', /ebs|managed disks?|persistent disk|block storage/],
        ['Object Storage', /amazons3|\bs3\b|blob|object storage|storage accounts?|cloud storage|\bgcs\b/],
        ['Database', /rds|database|cosmos|dynamodb|spanner|cloud sql/],
        ['Cache', /redis|cache|memorydb/], ['Container', /kubernetes|container|\baks\b|amazoneks|\bgke\b|amazonecs|fargate/],
        ['Networking', /vpc|load.?balanc|network|cdn|front door|\bnat\b|gateway|public ip/],
        ['Serverless', /lambda|functions|app service|cloud run/], ['Compute/VM', /ec2|virtual machine|compute engine|\bvm\b|vmss/],
    ];
    const serviceName = raw => SERVICE_RULES.find(([, re]) => re.test(raw.toLowerCase()))?.[0] || raw || 'UnknownService';
    function group(map, name, cost) {
        if (!map.has(name)) map.set(name, {cost: zero(), rows: 0});
        const value = map.get(name); add(value.cost, cost); value.rows++; return value;
    }
    function list(map, key, total) {
        return [...map].map(([name, v]) => ({[key]: name, BilledCost: rounded(v.cost), rows: v.rows, pctOfTotal: total ? amount(v.cost) / total * 100 : 0}))
            .sort((a, b) => b.BilledCost - a.BilledCost);
    }

    function validate(files, configs, assumedCurrency = '') {
        const errors = [], sources = [], warnings = new Set();
        let errorCount = 0, rowCount = 0, unknownProviders = 0;
        const fail = (file, line, column, message) => { errorCount++; if (errors.length < LIMITS.errors) errors.push({file, line, column, message}); };
        const assumed = assumedCurrency.trim().toUpperCase();
        if (assumed && !/^[A-Z]{3}$/.test(assumed)) fail('', null, 'Currency assumption', 'Use a three-letter currency code such as USD.');
        if (!files.length || files.length > LIMITS.files) fail('', null, '', `Choose between 1 and ${LIMITS.files} files.`);
        const hashes = new Set(), currencies = new Set(), bases = new Set();
        const total = zero(), unProduct = zero(), unPlatform = zero(), unEnvironment = zero();
        const products = new Map(), services = new Map(), environments = new Map(), platforms = new Map(), providers = new Map(), resources = new Map();
        const inventory = new Map(), unidentified = {cost: zero(), rows: 0};
        for (let index = 0; index < files.length; index++) {
            const file = files[index], config = configs[index];
            if (hashes.has(file.sha256)) { fail(file.name, null, '', 'This file has identical contents to another selected file. Remove one copy.'); continue; }
            hashes.add(file.sha256);
            if (!config || !Object.hasOwn(FORMATS, config.format) || !['billed', 'effective'].includes(config.costBasis) || !['aws', 'azure', 'gcp', 'unknown'].includes(config.provider)) { fail(file.name, null, '', 'Choose a valid format, provider, and cost basis.'); continue; }
            const mapping = config.mapping || {};
            if (!mapping.cost) { fail(file.name, null, 'Cost amount', 'Choose the column containing the cost amount.'); continue; }
            const invalidMapping = Object.entries(mapping).filter(([, value]) => value && !file.headers.includes(value));
            if (invalidMapping.length) { fail(file.name, null, '', 'A mapped column no longer exists. Review the mappings.'); continue; }
            const indices = Object.fromEntries(Object.keys(FIELDS).map(k => [k, file.headers.indexOf(mapping[k])]));
            const source = {file: file.name, sha256: file.sha256, schema: config.format, costColumn: mapping.cost, costBasis: config.costBasis,
                rows: 0, currencies: new Set(), missingCurrencyRows: 0, assumedCurrencyRows: 0, invalidDateFields: 0, invalidTagRows: 0,
                usagePeriod: {start: null, end: null}, billingPeriod: {start: null, end: null}, incompleteUsage: false, incompleteBilling: false,
                columnMapping: {...mapping}, providerFallback: config.provider, delimiter: file.delimiter};
            const sourceTotal = zero(); bases.add(config.costBasis);
            if (config.format === 'gcp') warnings.add('GCP totals use the mapped cost column. Credits stored in separate fields are not applied automatically; use a net-cost column if needed.');
            for (const record of file.records) {
                rowCount++;
                if (rowCount > LIMITS.rows) { fail('', null, '', `Use at most ${LIMITS.rows.toLocaleString()} rows across all selected files.`); break; }
                if (record.cells.length !== file.headers.length) { fail(file.name, record.line, '', `Expected ${file.headers.length} columns but found ${record.cells.length}. Check the delimiter and CSV quoting.`); continue; }
                const get = key => indices[key] >= 0 ? record.cells[indices[key]].trim() : '';
                let cost;
                try { cost = decimal(get('cost')); amount(cost); } catch (e) { fail(file.name, record.line, mapping.cost, e.message); continue; }
                const explicitCurrency = get('currency').toUpperCase(), currency = explicitCurrency || assumed;
                if (currency && !/^[A-Z]{3}$/.test(currency)) { fail(file.name, record.line, mapping.currency || 'Currency', 'Use a three-letter billing currency code.'); continue; }
                if (currency) { currencies.add(currency); source.currencies.add(currency); }
                else source.missingCurrencyRows++;
                if (!explicitCurrency && assumed) source.assumedCurrencyRows++;
                let tags = {};
                try { tags = tagsOf(get('tags')); } catch { source.invalidTagRows++; }
                const product = clean(get('product')) || tags.product || 'UnTagged';
                const platform = clean(get('platform')) || tags.platform || 'unknown';
                const envRaw = clean(get('environment')) || tags.environment || '';
                const environment = ({production: 'prod', development: 'dev', testing: 'test', staging: 'stage'})[envRaw.toLowerCase()] || envRaw.toLowerCase() || 'UnTagged';
                const provider = providerName(get('provider') || config.provider);
                if (provider === 'unknown') unknownProviders++;
                const service = serviceName(get('service'));
                if (!get('service')) warnings.add('Some rows have no service value and are grouped as UnknownService.');
                source.rows++; add(sourceTotal, cost); add(total, cost);
                if (product === 'UnTagged') add(unProduct, cost);
                if (platform === 'unknown') add(unPlatform, cost);
                if (environment === 'UnTagged') add(unEnvironment, cost);
                const productGroup = group(products, product, cost), platformGroup = group(platforms, platform, cost);
                platformGroup.products ??= new Map();
                const localProduct = group(platformGroup.products, product, cost);
                for (const parent of [productGroup, platformGroup, localProduct]) {
                    parent.services ??= new Map(); parent.environments ??= new Map(); parent.providers ??= new Map();
                    group(parent.services, service, cost); group(parent.environments, environment, cost); group(parent.providers, provider, cost);
                }
                group(services, service, cost); group(environments, environment, cost); group(providers, provider, cost);
                if (get('resource')) group(resources, get('resource'), cost);
                if (get('resource')) {
                    const identity = [provider, get('account'), get('region'), get('zone'), get('resource')];
                    const key = JSON.stringify(identity);
                    if (!inventory.has(key)) inventory.set(key, {provider, account: identity[1], region: identity[2], zone: identity[3], resourceId: identity[4], cost: zero(), rows: 0, services: new Set(), sourceHashes: new Set()});
                    const item = inventory.get(key); add(item.cost, cost); item.rows++; item.services.add(get('service')); item.sourceHashes.add(file.sha256);
                } else { add(unidentified.cost, cost); unidentified.rows++; }
                for (const [key, period, edge] of [['usage_start', 'usagePeriod', 'start'], ['usage_end', 'usagePeriod', 'end'], ['billing_start', 'billingPeriod', 'start'], ['billing_end', 'billingPeriod', 'end']]) {
                    const raw = get(key), date = dateOf(raw);
                    if (!date) { source.invalidDateFields++; source[period === 'usagePeriod' ? 'incompleteUsage' : 'incompleteBilling'] = true; continue; }
                    const current = source[period][edge];
                    if (!current || (edge === 'start' ? date < current : date > current)) source[period][edge] = date;
                }
                for (const [start, end] of [['usage_start', 'usage_end'], ['billing_start', 'billing_end']]) {
                    const a = dateOf(get(start)), b = dateOf(get(end));
                    if (a && b && a > b) fail(file.name, record.line, mapping[end], 'The period end is before its start.');
                }
            }
            source.currencies = [...source.currencies].sort(); source.sourceCostTotal = decimalText(sourceTotal);
            sources.push(source);
        }
        const missing = sources.reduce((n, s) => n + s.missingCurrencyRows, 0);
        if (currencies.size > 1) fail('', null, 'Billing currency', `Cannot combine ${[...currencies].join(', ')}. Import each currency separately; currency conversion is not supported.`);
        if (currencies.size && missing) fail('', null, 'Billing currency', 'Some rows have no currency. Set an explicit assumption only if you know their currency, or import them separately.');
        if (bases.size > 1) fail('', null, 'Cost basis', 'Billed and effective costs cannot be combined. Import each cost basis separately.');
        if (missing) warnings.add('Currency is unspecified. Amounts have no currency symbol; cross-file monetary comparability has not been verified.');
        if (sources.some(s => s.assumedCurrencyRows)) warnings.add(`Missing currencies were filled using your ${assumed} assumption. Explicit currencies were preserved.`);
        if (sources.some(s => s.invalidTagRows)) warnings.add('Some tags could not be parsed. Review the unallocated spend and map product/platform columns where available.');
        if (sources.some(s => s.invalidDateFields)) warnings.add('Some period fields are missing or invalid. Observed date ranges are incomplete.');
        if (unknownProviders) warnings.add(`${unknownProviders} rows have an unknown provider. Review provider mapping before generating infrastructure templates.`);
        warnings.add('Browser import summarizes cost and explicit tags. Hardware sizing, resource-name inference, and utilization are not analyzed.');
        if (files.length > 1) warnings.add('Only identical whole files are detected as duplicates. Check that exports do not contain overlapping billing rows.');
        if (!rowCount) fail('', null, '', 'No billing rows were available for validation.');
        let totalNumber = 0, reportTotal = 0;
        try { totalNumber = amount(total); reportTotal = rounded(total); } catch (e) { fail('', null, 'Cost', e.message); }
        if (errorCount) return {ok: false, errors, errorCount, warnings: [...warnings], report: null, rowCount};
        const pct = (part, whole) => whole ? part / whole * 100 : 0;
        function productList(map, whole) {
            return list(map, 'Product', whole).map(p => { const g = map.get(p.Product); return {...p, pctOfPlatform: pct(p.BilledCost, whole),
                services: list(g.services, 'ServiceCategory', p.BilledCost), environments: list(g.environments, 'Environment', p.BilledCost),
                providers: list(g.providers, 'ProviderName', p.BilledCost), serviceCount: g.services.size, vmSkus: [], ec2Skus: []}; });
        }
        let report;
        try {
            const periods = Object.fromEntries(['usagePeriod', 'billingPeriod'].map(key => {
                const starts = sources.map(s => s[key].start).filter(Boolean).sort(), ends = sources.map(s => s[key].end).filter(Boolean).sort();
                return [key, {start: starts[0] || null, end: ends.at(-1) || null, complete: starts.length === sources.length && ends.length === sources.length && !sources.some(s => s[key === 'usagePeriod' ? 'incompleteUsage' : 'incompleteBilling'])}];
            }));
            report = {sampleData: false, TotalBilledCost: reportTotal,
                analysis: {schemaVersion: 1, generatedAt: new Date().toISOString(), importMethod: 'browser', currency: [...currencies][0] || null,
                    costBasis: [...bases][0], sourceCount: sources.length, rowCount, sources, ...periods,
                    reconciliation: {sourceTotal: totalNumber, reportTotal, difference: Number((reportTotal - totalNumber).toFixed(8))}, warnings: [...warnings]},
                allocation: {unallocatedProductCost: rounded(unProduct), unallocatedPlatformCost: rounded(unPlatform), unallocatedEnvironmentCost: rounded(unEnvironment)},
                products: productList(products, totalNumber), services: list(services, 'ServiceCategory', totalNumber), environments: list(environments, 'Environment', totalNumber), providers: list(providers, 'ProviderName', totalNumber),
                platforms: list(platforms, 'Platform', totalNumber).map(p => { const g = platforms.get(p.Platform); return {...p, products: productList(g.products, p.BilledCost), services: list(g.services, 'ServiceCategory', p.BilledCost), vmSkus: [], ec2Skus: []}; }),
                resourceNames: list(resources, 'ResourceName', totalNumber), namingConvention: {source: 'Resource identifiers as supplied; no naming inference', mode: 'explicit allocation only', pattern: []},
                resourceInventory: [...inventory.values()].map(({cost, services, sourceHashes, ...item}) => ({...item, cost: rounded(cost), services: [...services], sourceHashes: [...sourceHashes]})),
                resourceInventorySummary: {unidentifiedCost: rounded(unidentified.cost), unidentifiedRows: unidentified.rows},
            };
        } catch (e) { fail('', null, 'Cost', e.message); return {ok: false, errors, errorCount, report: null, warnings: [...warnings], rowCount}; }
        return {ok: true, errors: [], errorCount: 0, warnings: [...warnings], report, rowCount};
    }
    const api = {LIMITS, FIELDS, FORMATS, parseCSV, detect, validate, dateOf};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.BillingImport = api;
})(globalThis);
