const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const adoption = require('../adoption.js');
const billing = require('../billing_import.js');
const aws = (extra = {}) => ({provider: 'aws', resourceId: 'i-0123456789abcdef0', account: '123456789012', region: 'us-east-1', cost: 10, sourceHashes: ['a'.repeat(64)], ...extra});
const gcp = (extra = {}) => ({provider: 'gcp', resourceId: '//compute.googleapis.com/projects/atlas-prod/zones/us-central1-a/instances/web-prod', cost: 4, ...extra});
const report = resources => ({sampleData: false, analysis: {currency: 'USD', costBasis: 'billed'}, resourceInventory: resources});

test('AWS ARN and explicit billing scope produce canonical identity; conflicts block', () => {
    const a = adoption.classify(aws()), b = adoption.classify(aws({resourceId: 'arn:aws:ec2:us-east-1:123456789012:instance/i-0123456789abcdef0', account: '', region: ''}));
    assert.equal(adoption.identity(a), adoption.identity(b));
    assert.equal(adoption.classify(aws({resourceId: b.resourceId, account: '234567890123'})).status, 'unresolved');
    assert.notEqual(adoption.identity(a), adoption.identity(adoption.classify(aws({account: '234567890123'}))));
});
test('scope defaults fill only blanks and never infer an AWS resource from a name', () => {
    assert.equal(adoption.classify(aws({account: '', region: ''})).status, 'unresolved');
    assert.equal(adoption.classify(aws({account: '', region: ''}), {awsAccount: '123456789012', awsRegion: 'us-east-1'}).status, 'candidate');
    assert.equal(adoption.classify(aws(), {awsAccount: '234567890123'}).account, '123456789012');
    assert.equal(adoption.classify(aws({resourceId: 'web-prod'})).status, 'unresolved');
});
test('GCP instances and disks use explicit project/zone and reject conflicting scope', () => {
    const instance = adoption.classify(gcp());
    assert.equal(instance.account, 'atlas-prod'); assert.equal(instance.zone, 'us-central1-a'); assert.equal(instance.region, 'us-central1');
    assert.equal(adoption.classify(gcp({account: 'other-project'})).status, 'unresolved');
    assert.equal(adoption.classify(gcp({resourceId: 'web-prod'})).status, 'unresolved');
    assert.equal(adoption.classify(gcp({resourceId: 'projects/atlas-prod/zones/us-central1-a/disks/data'})).resourceType, 'google_compute_disk');
});
test('aliases merge, identical resource IDs in different accounts stay separate, unsupported costs remain visible', () => {
    const value = report([aws(), aws({resourceId: 'arn:aws:ec2:us-east-1:123456789012:instance/i-0123456789abcdef0'}), aws({account: '234567890123'}), aws({resourceId: 'other', cost: -3})]);
    const rows = adoption.candidates(value);
    assert.equal(rows.length, 3); assert.equal(rows[0].cost, 20); assert.equal(rows.reduce((sum, r) => sum + r.cost, 0), 27);
    assert.equal(rows.at(-1).status, 'unresolved');
});
test('packages require current inventory, real data, eligible identities and bounded batches', () => {
    assert.throws(() => adoption.candidates({resourceNames: []}), /no scoped/);
    const value = report([aws()]), rows = adoption.candidates(value);
    assert.equal(adoption.manifest(value, rows).resources[0].resourceId, aws().resourceId);
    assert.throws(() => adoption.manifest({...value, sampleData: true}, rows), /Sample/);
    assert.throws(() => adoption.manifest(value, []), /1–200/);
    assert.throws(() => adoption.manifest(value, [...rows, ...rows]), /once/);
    assert.throws(() => adoption.manifest(value, Array(201).fill(rows[0])), /1–200/);
    assert.throws(() => adoption.manifest(value, [{...rows[0], discovery: {status: 'imported'}}]), /unmanaged/);
});
test('allowlisted discovery results match scoped identities and reject raw state/duplicate results', () => {
    const items = adoption.candidates(report([aws(), aws({account: '234567890123'})]));
    const inventory = {schemaVersion: 1, kind: 'atlas-adoption-inventory', resources: [{...aws(), status: 'imported', message: 'Clean', configuration: {secret: 'NEVER'}}]};
    const rows = adoption.results(inventory, items);
    assert.equal(rows.filter(r => r.discovery).length, 1); assert.equal(rows[0].discovery.status, 'imported');
    assert.doesNotMatch(JSON.stringify(rows), /NEVER/);
    assert.throws(() => adoption.results({version: 4, resources: []}, items), /Raw Terraform/);
    assert.throws(() => adoption.results({...inventory, resources: [...inventory.resources, ...inventory.resources]}, items), /duplicate/);
});
test('billing importer preserves scoped IDs, independent account costs and missing-ID costs', () => {
    const csv = 'lineItem/UnblendedCost,lineItem/ResourceId,lineItem/UsageAccountId,product/regionCode,lineItem/ProductCode,lineItem/CurrencyCode\n10,i-0123456789abcdef0,123456789012,us-east-1,AmazonEC2,USD\n2,i-0123456789abcdef0,234567890123,us-east-1,AmazonEC2,USD\n-1,,123456789012,us-east-1,AmazonEC2,USD\n';
    const parsed = billing.parseCSV(csv), config = billing.detect(parsed.headers);
    assert.equal(config.mapping.account, 'lineItem/UsageAccountId'); assert.equal(config.mapping.region, 'product/regionCode');
    const result = billing.validate([{...parsed, name: 'billing.csv', sha256: 'a'.repeat(64), delimiter: ','}], [config]);
    assert.equal(result.ok, true); assert.equal(result.report.resourceInventory.length, 2); assert.equal(result.report.resourceInventorySummary.unidentifiedCost, -1);
    assert.equal(adoption.candidates(result.report).filter(r => r.status === 'candidate').length, 2);
});
test('ZIP package is a valid extractable archive with exact tool contents and no path traversal', async () => {
    const tool = fs.readFileSync(path.join(__dirname, '..', 'adoption_tool.py'), 'utf8');
    const blob = adoption.zip({'adoption_tool.py': tool, 'candidates.json': JSON.stringify(adoption.manifest(report([aws()]), adoption.candidates(report([aws()])))), 'README.md': 'Resources → IaC'});
    const bytes = Buffer.from(await blob.arrayBuffer());
    assert.equal(bytes.readUInt32LE(0), 0x04034b50);
    const filenameLength = bytes.readUInt16LE(26), dataLength = bytes.readUInt32LE(18);
    assert.equal(bytes.subarray(30, 30 + filenameLength).toString(), 'adoption_tool.py');
    assert.equal(bytes.subarray(30 + filenameLength, 30 + filenameLength + dataLength).toString(), tool);
    assert.equal(bytes.readUInt32LE(bytes.length - 22), 0x06054b50);
    assert.equal(bytes.readUInt16LE(bytes.length - 12), 3);
    assert.throws(() => adoption.zip({'../bad': 'text'}), /relative/);
});
