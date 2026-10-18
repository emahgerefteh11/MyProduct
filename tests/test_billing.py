import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cur_to_json as engine
from terraform_generator import detect_stack_provider
from generate_dashboard import inject_data


class BillingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def write_csv(self, rows, name='billing.csv', bom=False):
        path = self.root / name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open('w', newline='', encoding='utf-8-sig' if bom else 'utf-8') as output:
            writer = csv.DictWriter(output, fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def aws(self, **overrides):
        return {
            'lineItem/UnblendedCost': '100.25', 'lineItem/ResourceId': 'i-123abc',
            'lineItem/ProductCode': 'AmazonEC2', 'lineItem/UsageType': 'BoxUsage:t3.medium',
            'lineItem/UsageAmount': '24', 'lineItem/UsageUnit': 'Hrs',
            'lineItem/CurrencyCode': 'USD', 'lineItem/UsageStartDate': '2026-09-01T00:00:00Z',
            'lineItem/UsageEndDate': '2026-09-02T00:00:00Z',
            'resourceTags/user:Product': 'Checkout', 'resourceTags/user:Environment': 'Production',
            'resourceTags/user:Platform': 'Commerce', **overrides,
        }

    def summarize(self, rows, **kwargs):
        return engine.build_summary(engine.load_cur(self.write_csv(rows), engine.DEFAULT_NAMING_CONVENTION, **kwargs))

    def test_native_aws_tags_and_provenance(self):
        report = self.summarize([self.aws()])
        self.assertEqual(report['products'][0]['Product'], 'Checkout')
        self.assertEqual(report['platforms'][0]['Platform'], 'Commerce')
        self.assertEqual(report['environments'][0]['Environment'], 'prod')
        self.assertEqual(report['analysis']['currency'], 'USD')
        self.assertEqual(report['analysis']['reconciliation']['difference'], 0)
        self.assertEqual(report['analysis']['usagePeriod'], {'start': '2026-09-01T00:00:00Z', 'end': '2026-09-02T00:00:00Z', 'complete': True})
        self.assertEqual(report['analysis']['sources'][0]['file'], 'billing.csv')
        self.assertEqual(len(report['analysis']['sources'][0]['sha256']), 64)
        self.assertTrue(report['analysis']['generatedAt'].endswith('Z'))

    def test_focus_provider_tags_and_billed_cost_precedence(self):
        for provider, normalized in [('Amazon Web Services', 'aws'), ('Microsoft Azure', 'azure'), ('Google Cloud', 'gcp')]:
            with self.subTest(provider=provider):
                report = self.summarize([{
                    'BilledCost': '50', 'EffectiveCost': '30', 'ProviderName': provider,
                    'ResourceId': 'i-123abc', 'ServiceName': 'AmazonEC2', 'BillingCurrency': 'EUR',
                    'Tags': json.dumps({'Product': 'Checkout', 'Environment': 'prod', 'Platform': 'Commerce'}),
                    'SubAccountId': 'account-fallback', 'ConsumedQuantity': '24', 'ConsumedUnit': 'Hours',
                    'PricingQuantity': '1000', 'PricingUnit': 'Requests',
                }])
                self.assertEqual(report['providers'][0]['ProviderName'], normalized)
                self.assertEqual(report['platforms'][0]['Platform'], 'Commerce')
                self.assertEqual(report['products'][0]['Product'], 'Checkout')
                self.assertEqual(report['TotalBilledCost'], 50)
                self.assertEqual(report['analysis']['costBasis'], 'billed')
                self.assertEqual(detect_stack_provider('Commerce', report['platforms'][0]['products'][0]), normalized)

    def test_mixed_provider_products_require_an_explicit_iac_choice(self):
        report = self.summarize([
            {'BilledCost': '10', 'ProviderName': provider, 'Tags': '{"product":"Checkout","platform":"Commerce"}'}
            for provider in ('AWS', 'Google Cloud')
        ])
        with self.assertRaisesRegex(ValueError, '--provider'):
            detect_stack_provider('Commerce', report['platforms'][0]['products'][0])

    def test_focus_provider_without_tags_and_account_fallback(self):
        report = self.summarize([{'BilledCost': '10', 'ProviderName': 'AWS', 'ResourceId': 'i-123abc', 'SubAccountName': 'Engineering'}])
        self.assertEqual(report['providers'][0]['ProviderName'], 'aws')
        self.assertEqual(report['platforms'][0]['Platform'], 'Engineering')
        self.assertEqual(report['products'][0]['Product'], 'UnTagged')

    def test_azure_json_tags(self):
        report = self.summarize([{
            'Cost': '25', 'InstanceId': '/subscriptions/demo/providers/Microsoft.Compute/virtualMachines/worker01',
            'ServiceName': 'Virtual Machines', 'Currency': 'GBP',
            'Tags': json.dumps({'Product': 'Checkout', 'Env': 'Production', 'Platform': 'Commerce'}),
        }])
        self.assertEqual(report['products'][0]['Product'], 'Checkout')
        self.assertEqual(report['platforms'][0]['Platform'], 'Commerce')
        self.assertEqual(report['environments'][0]['Environment'], 'prod')
        self.assertEqual(report['analysis']['currency'], 'GBP')

    def test_gcp_label_formats(self):
        for labels in ['project:labels:{"product":"Checkout","environment":"dev","platform":"Commerce"}', '[{"key":"product","value":"Checkout"},{"key":"platform","value":"Commerce"}]']:
            report = self.summarize([{'Cost': '25', 'LineItem': '1234', 'UsageStartDate': '2026-09-01', 'ServiceDescription': 'Compute Engine', 'Labels': labels}])
            self.assertEqual(report['products'][0]['Product'], 'Checkout')
            self.assertEqual(report['platforms'][0]['Platform'], 'Commerce')

    def test_opaque_ids_do_not_become_ownership(self):
        report = self.summarize([self.aws(**{'resourceTags/user:Product': '', 'resourceTags/user:Platform': '', 'resourceTags/user:Environment': ''})])
        self.assertEqual(report['products'][0]['Product'], 'UnTagged')
        self.assertEqual(report['platforms'][0]['Platform'], 'unknown')
        self.assertEqual(report['allocation']['unallocatedProductCost'], 100.25)

    def test_provider_specific_naming_and_precedence(self):
        config = {'default': {'pattern': ['resource_type', 'product', 'environment']}, 'providers': {'aws': {'pattern': ['resource_type', 'platform', 'product', 'environment', 'instance']}}}
        path = self.write_csv([self.aws(**{'lineItem/ResourceId': 'vm-Engineering-Orders-dev-001'})])
        fallback = engine.build_summary(engine.load_cur(path, config))
        override = engine.build_summary(engine.load_cur(path, config, 'override'))
        self.assertEqual(fallback['products'][0]['Product'], 'Checkout')
        self.assertEqual(override['products'][0]['Product'], 'Orders')
        self.assertEqual(override['platforms'][0]['Platform'], 'Engineering')

    def test_missing_currency_is_not_assumed(self):
        row = self.aws(**{'lineItem/CurrencyCode': ''})
        report = self.summarize([row])
        self.assertIsNone(report['analysis']['currency'])
        self.assertTrue(report['analysis']['warnings'])
        report = self.summarize([row], assumed_currency='eur')
        self.assertEqual(report['analysis']['currency'], 'EUR')
        self.assertEqual(report['analysis']['sources'][0]['assumedCurrencyRows'], 1)

    def test_currency_assumption_does_not_overwrite_explicit_currency(self):
        report = self.summarize([self.aws()], assumed_currency='EUR')
        self.assertEqual(report['analysis']['currency'], 'USD')
        with self.assertRaisesRegex(ValueError, 'different billing currencies'):
            self.summarize([self.aws(), self.aws(**{'lineItem/CurrencyCode': ''})], assumed_currency='EUR')

    def test_mixed_currencies_and_partial_unknown_rejected(self):
        for code in ['EUR', '']:
            with self.subTest(currency=code), self.assertRaises(ValueError):
                self.summarize([self.aws(), self.aws(**{'lineItem/CurrencyCode': code})])

    def test_invalid_amounts_are_not_silently_counted(self):
        for value in ['', 'NaN', 'Infinity', '-Infinity', '1e999', 'not a number']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'row 2'):
                self.summarize([self.aws(**{'lineItem/UnblendedCost': value})])

    def test_negative_credits_reconcile(self):
        report = self.summarize([self.aws(), self.aws(**{'lineItem/UnblendedCost': '-10.25'})])
        self.assertEqual(report['TotalBilledCost'], 90)
        self.assertEqual(report['analysis']['reconciliation']['difference'], 0)

    def test_bom_and_malformed_optional_metadata(self):
        path = self.write_csv([{'BilledCost': '10', 'ProviderName': 'AWS', 'Tags': '{invalid', 'ChargePeriodStart': 'not a date'}], bom=True)
        report = engine.build_summary(engine.load_cur(path, engine.DEFAULT_NAMING_CONVENTION))
        self.assertEqual(report['TotalBilledCost'], 10)
        self.assertEqual(report['analysis']['sources'][0]['invalidTagRows'], 1)
        self.assertEqual(report['analysis']['sources'][0]['invalidDateFields'], 1)
        self.assertFalse(report['analysis']['usagePeriod']['complete'])

    def test_billed_and_effective_costs_cannot_be_combined(self):
        paths = [self.write_csv([{'BilledCost': '10'}], 'billed.csv'), self.write_csv([{'EffectiveCost': '8'}], 'effective.csv')]
        states = [engine.load_cur(path, engine.DEFAULT_NAMING_CONVENTION) for path in paths]
        self.assertEqual(engine.build_summary(states[1])['analysis']['costBasis'], 'effective')
        with self.assertRaisesRegex(ValueError, 'billed and effective'):
            engine.build_summary(engine.merge_cur_states(states))

    def test_duplicate_cli_inputs_preserve_existing_output(self):
        first = self.write_csv([self.aws()])
        copied = self.root / 'renamed.csv'
        copied.write_bytes(first.read_bytes())
        output = self.root / 'report.json'
        output.write_text('existing report')
        result = subprocess.run([sys.executable, str(ROOT / 'cur_to_json.py'), '--input', str(first), '--input', str(copied), '--output', str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Duplicate billing input', result.stderr)
        self.assertEqual(output.read_text(), 'existing report')

    def test_mock_multicloud_totals_reconcile(self):
        files = ['mock_aws_cur_iaas_heavy_named20251122.csv', 'mock_gcp_billing.csv']
        config = engine.load_naming_convention(ROOT / 'naming_convention.sample.json')
        states = [engine.load_cur(ROOT / name, config, assumed_currency='USD') for name in files]
        report = engine.build_summary(engine.merge_cur_states(states))
        self.assertEqual(report['TotalBilledCost'], 233951.95)
        self.assertEqual(report['analysis']['rowCount'], 1203)
        self.assertEqual(report['analysis']['sourceCount'], 2)
        self.assertEqual(report['analysis']['reconciliation']['difference'], 0)

    def test_demo_generation_escapes_script_terminators(self):
        template = self.root / 'template.html'
        template.write_text('<script>let reportData = null;</script>')
        data = self.root / 'data.json'
        data.write_text(json.dumps({'label': '</script><script>alert(1)</script>'}))
        output = self.root / 'demo.html'
        inject_data(template, data, output)
        self.assertEqual(output.read_text().count('</script>'), 1)
        self.assertIn('\\u003c/script>', output.read_text())

    def test_demo_generation_requires_valid_anchor(self):
        template = self.root / 'template.html'
        template.write_text('<html></html>')
        data = self.root / 'data.json'
        data.write_text('{}')
        output = self.root / 'demo.html'
        with self.assertRaisesRegex(ValueError, 'anchor'):
            inject_data(template, data, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
