const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const vm = require('node:vm');
const B = require('../billing_import.js');
const root = path.join(__dirname, '..');

function source(text, name = 'billing.csv', delimiter = ',') {
    return {name, delimiter, sha256: crypto.createHash('sha256').update(text).digest('hex'), ...B.parseCSV(text, delimiter)};
}
function check(text, {currency = '', config, delimiter = ','} = {}) {
    const file = source(text, 'billing.csv', delimiter);
    return B.validate([file], [config || B.detect(file.headers)], currency);
}
const csv = values => values.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(',');

test('CSV supports BOM, CRLF, quoted delimiters, escaped quotes and physical line numbers', () => {
    const result = B.parseCSV('\uFEFFCost,Product\r\n1,"Name, with ""quotes"""\r\n2,"two\nlines"\r\n3,last\r\n');
    assert.deepEqual(result.headers, ['Cost', 'Product']);
    assert.equal(result.records[0].cells[1], 'Name, with "quotes"');
    assert.equal(result.records[1].cells[1], 'two\nlines');
    assert.equal(result.records[2].line, 5);
});
test('CSV rejects empty, ambiguous, and malformed inputs', () => {
    for (const text of ['', 'Cost,Product\n', 'Cost,cost\n1,2', ',Product\n1,x', 'Cost,Product\n1,"x', 'Cost,Product\n1,"x"oops', 'Cost,Product\n1,x"y']) assert.throws(() => B.parseCSV(text));
    assert.equal(check('Cost;Product\n12;app', {delimiter: ';'}).report.TotalBilledCost, 12);
    assert.equal(check('Cost\tProduct\n12\tapp', {delimiter: '\t'}).report.products[0].Product, 'app');
});
test('detects AWS, Azure, GCP, FOCUS and custom headers', () => {
    assert.equal(B.detect(['lineItem/UnblendedCost']).format, 'aws');
    assert.equal(B.detect(['CostInBillingCurrency', 'ResourceId']).format, 'azure');
    assert.equal(B.detect(['cost', 'service.description']).format, 'gcp');
    assert.equal(B.detect(['EffectiveCost', 'ProviderName']).costBasis, 'effective');
    assert.equal(B.detect(['Amount', 'Team']).format, 'custom');
});
test('AWS native tags, provider, decimal costs, and credits reach the report', () => {
    const result = check('lineItem/UnblendedCost,lineItem/ProductCode,resourceTags/user:Product,resourceTags/user:Platform,resourceTags/user:Environment,lineItem/CurrencyCode\n0.1,AmazonEC2,checkout,commerce,production,USD\n0.2,AmazonEC2,checkout,commerce,production,USD\n-0.05,AmazonS3,checkout,commerce,production,USD');
    assert.equal(result.ok, true);
    assert.equal(result.report.TotalBilledCost, .25);
    assert.equal(result.report.analysis.sources[0].sourceCostTotal, '0.25');
    assert.equal(result.report.platforms[0].products[0].providers[0].ProviderName, 'aws');
    assert.equal(result.report.environments[0].Environment, 'prod');
    assert.equal(result.report.services[0].ServiceCategory, 'Compute/VM');
    assert.equal(result.report.allocation.unallocatedProductCost, 0);
    assert.equal(result.report.sampleData, false);
});
test('Azure tags and GCP key/value label arrays are attributed without name inference', () => {
    const azure = check('Cost,ResourceId,Tags\n' + csv([2, '/subscriptions/id/resourceGroups/team/providers/Microsoft.Compute/virtualMachines/vm', JSON.stringify({Product: 'api', Platform: 'core'})]));
    assert.equal(azure.report.products[0].Product, 'api');
    assert.equal(azure.report.providers[0].ProviderName, 'azure');
    assert.equal(azure.report.platforms[0].Platform, 'core');
    const gcp = check('Cost,ServiceDescription,Labels\n' + csv([3, 'Cloud Storage', 'project:labels:' + JSON.stringify([{key:'product', value:'media'}])]));
    assert.equal(gcp.report.products[0].Product, 'media');
    assert.equal(gcp.report.services[0].ServiceCategory, 'Object Storage');
    assert.match(gcp.warnings.join(' '), /Credits.*not applied/);
});
test('FOCUS reads provider values and preserves effective-only cost basis', () => {
    const result = check('EffectiveCost,ProviderName,ServiceCategory,BillingCurrency\n42,Google Cloud,Database,EUR');
    assert.equal(result.report.analysis.costBasis, 'effective');
    assert.equal(result.report.providers[0].ProviderName, 'gcp');
    assert.equal(result.report.analysis.currency, 'EUR');
    assert.equal(result.report.analysis.sources[0].costColumn, 'EffectiveCost');
});
test('custom mappings change the amount and allocation used in the preview', () => {
    const file = source('Charge,Owner,Group,Money\n19.95,orders,retail,GBP');
    const config = B.detect(file.headers);
    assert.equal(B.validate([file], [config]).ok, false);
    Object.assign(config.mapping, {cost:'Charge', product:'Owner', platform:'Group', currency:'Money'});
    config.provider = 'aws';
    const result = B.validate([file], [config]);
    assert.equal(result.report.TotalBilledCost, 19.95);
    assert.equal(result.report.products[0].Product, 'orders');
    assert.equal(result.report.analysis.sources[0].columnMapping.platform, 'Group');
});
test('invalid and non-finite costs include filename, physical row and column, and block report creation', () => {
    for (const value of ['NaN', 'Infinity', '1e999', '$20', '1,000', '', '9007199254740991']) {
        const result = check('Cost,Product\n' + csv([value, 'api']));
        assert.equal(result.ok, false, value);
        assert.equal(result.report, null);
        assert.equal(result.errors[0].file, 'billing.csv');
        assert.equal(result.errors[0].line, 2);
        assert.equal(result.errors[0].column, 'Cost');
    }
});
test('row-width mismatches and backwards periods prevent acceptance', () => {
    assert.match(check('Cost,Product\n1,x,extra').errors[0].message, /Expected 2 columns/);
    assert.match(check('Cost,Product\n1').errors[0].message, /found 1/);
    const result = check('Cost,UsageStartDate,UsageEndDate\n1,2026-02-02,2026-02-01');
    assert.equal(result.ok, false);
    assert.match(result.errors[0].message, /end is before/);
});
test('known and unknown currencies only combine with an explicit compatible assumption', () => {
    const text = 'Cost,Currency\n1,USD\n2,';
    assert.equal(check(text).ok, false);
    assert.equal(check(text, {currency:'USD'}).report.TotalBilledCost, 3);
    assert.equal(check(text, {currency:'EUR'}).ok, false);
    assert.equal(check('Cost,Currency\n1,EUR', {currency:'USD'}).report.analysis.currency, 'EUR');
    assert.equal(check('Cost\n1').report.analysis.currency, null);
    assert.equal(check('Cost\n1', {currency:'dollars'}).ok, false);
    assert.equal(check('Cost,Currency\n1,US$').ok, false);
});
test('different source currencies, mixed cost bases and renamed duplicate files are rejected', () => {
    const a = source('BilledCost,BillingCurrency\n1,USD', 'a.csv');
    const b = source('EffectiveCost,BillingCurrency\n2,USD', 'b.csv');
    assert.equal(B.validate([a,b], [B.detect(a.headers),B.detect(b.headers)]).ok, false);
    const duplicate = {...a, name:'renamed.csv'};
    assert.match(B.validate([a,duplicate], [B.detect(a.headers),B.detect(a.headers)]).errors[0].message, /identical contents/);
    assert.equal(check('Cost,Currency\n1,USD\n1,GBP').ok, false);
});
test('unallocated dimensions, unknown providers and malformed tags remain visible', () => {
    const result = check('Cost,Tags,Provider,ResourceId\n' + csv([10,'{invalid','unrecognized-cloud','resource-myteam-myapp-prod-001']));
    assert.equal(result.report.products[0].Product, 'UnTagged');
    assert.equal(result.report.allocation.unallocatedProductCost, 10);
    assert.equal(result.report.allocation.unallocatedPlatformCost, 10);
    assert.equal(result.report.providers[0].ProviderName, 'unknown');
    assert.match(result.warnings.join(' '), /tags could not be parsed/);
    assert.match(result.warnings.join(' '), /unknown provider/);
    assert.equal(result.report.resourceNames[0].ResourceName, 'resource-myteam-myapp-prod-001');
});
test('missing or invalid dates warn; observed periods never claim missing coverage', () => {
    assert.equal(B.dateOf('2026-02-30'), null);
    assert.equal(B.dateOf('02/28/2026'), '2026-02-28T00:00:00.000Z');
    assert.equal(B.dateOf('not a date'), null);
    const partial = check('Cost,UsageStartDate,UsageEndDate\n1,2026-01-01,2026-01-02\n1,,2026-01-03');
    assert.equal(partial.report.analysis.usagePeriod.complete, false);
    const complete = check('Cost,UsageStartDate,UsageEndDate\n1,2026-01-01,2026-01-02');
    assert.equal(complete.report.analysis.usagePeriod.complete, true);
    assert.equal(complete.report.analysis.billingPeriod.complete, false);
});
test('sub-cent decimals accumulate before rounding and negative credits remain intact', () => {
    const result = check('Cost\n0.004\n0.004\n0.004\n-0.001');
    assert.equal(result.report.analysis.reconciliation.sourceTotal, .011);
    assert.equal(result.report.TotalBilledCost, .01);
    assert.equal(result.report.analysis.reconciliation.difference, -.001);
    assert.equal(check('Cost\n-0.005').report.TotalBilledCost, -.01);
});
test('hostile keys are plain data, and validation output is JSON serializable', () => {
    const result = check('Cost,Product,Platform\n1,__proto__,constructor\n2,<img src=x onerror=alert(1)>,constructor');
    assert.equal(result.report.TotalBilledCost, 3);
    assert.equal(result.report.platforms[0].Platform, 'constructor');
    assert.doesNotThrow(() => JSON.stringify(result));
    assert.equal({}.polluted, undefined);
});
test('large cancelling amounts cannot hide an unsafe group total', () => {
    const result = check('Cost,Product\n90000000000000,a\n90000000000000,a\n-90000000000000,b\n-90000000000000,b');
    assert.equal(result.ok, false);
    assert.match(result.errors[0].message, /too large/);
});
test('multiple files match the existing AWS/GCP fixture total', () => {
    const files = ['mock_aws_cur_iaas_heavy_named20251122.csv', 'mock_gcp_billing.csv'].map(name => source(fs.readFileSync(path.join(root,name),'utf8'),name));
    const result = B.validate(files, files.map(f => B.detect(f.headers)), 'USD');
    assert.equal(result.ok, true);
    assert.equal(result.report.TotalBilledCost, 233951.95);
    assert.equal(result.report.analysis.rowCount, 1203);
    assert.equal(result.report.analysis.sources[0].sha256.length, 64);
    assert.equal(result.report.analysis.sourceCount, 2);
    assert.match(result.warnings.join(' '), /overlapping billing rows/);
});
test('validation bounds error output without hiding the total failure count', () => {
    const result = check('Cost,Product\n' + Array(120).fill('nope,app').join('\n'));
    assert.equal(result.errorCount, 120);
    assert.equal(result.errors.length, 100);
    assert.equal(result.report, null);
});
test('worker reads original bytes, fingerprints sources, and rejects corrupt or oversized files', async () => {
    const messages = [];
    const context = vm.createContext({BillingImport:B, TextDecoder, crypto:crypto.webcrypto, importScripts(){}, self:{postMessage:m=>messages.push(m)}});
    vm.runInContext(fs.readFileSync(path.join(root,'billing_worker.js'),'utf8'),context);
    const content = Buffer.from('\uFEFFCost,Product\n1,app');
    const file = {name:'a.csv',size:content.length,arrayBuffer:async()=>content};
    await context.self.onmessage({data:{id:1,type:'prepare',files:[file],delimiter:','}});
    assert.equal(messages[0].result[0].rows, 1);
    await context.self.onmessage({data:{id:2,type:'validate',configs:[messages[0].result[0].detected],currency:'USD'}});
    assert.equal(messages[1].result.report.analysis.sources[0].sha256, crypto.createHash('sha256').update(content).digest('hex'));
    await context.self.onmessage({data:{id:3,type:'prepare',files:[{...file,size:B.LIMITS.bytes+1}],delimiter:','}});
    assert.match(messages[2].error,/20 MiB/);
    await context.self.onmessage({data:{id:4,type:'validate',configs:[],currency:''}});
    assert.match(messages[3].error,/first/);
    await context.self.onmessage({data:{id:5,type:'prepare',files:[{...file,arrayBuffer:async()=>Buffer.from([0xff])}],delimiter:','}});
    assert.match(messages[4].error,/a.csv/);
});
