import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adoption_tool as adoption
import cur_to_json

ACCOUNT = "123456789012"
INSTANCE = "i-0123456789abcdef0"
VOLUME = "vol-0123456789abcdef0"


def aws(rid=INSTANCE, account=ACCOUNT):
    return {"provider": "aws", "account": account, "region": "us-east-1", "resourceId": rid, "cost": 10.25, "sourceHashes": ["a" * 64]}


def gcp(kind="instances", name="web-prod"):
    return {"provider": "gcp", "resourceId": f"projects/atlas-prod/zones/us-central1-a/{kind}/{name}", "cost": 4}


def plan(resources, action=None):
    return {"format_version": "1.2", "complete": True, "resource_changes": [{"address": r["address"], "change": {"actions": action or ["no-op"], "importing": {"id": r["importId"]}}} for r in resources]}


class IdentityTests(unittest.TestCase):
    def test_aws_arn_and_bare_id_have_same_address(self):
        bare = adoption.classify(aws())
        arn = adoption.classify({"provider": "aws", "resourceId": f"arn:aws:ec2:us-east-1:{ACCOUNT}:instance/{INSTANCE}"})
        self.assertEqual(bare["address"], arn["address"])
        self.assertNotEqual(bare["address"], adoption.classify(aws(account="234567890123"))["address"])

    def test_missing_conflicting_and_malicious_scope_rejected(self):
        for value in [aws(account=""), aws(rid="web-prod"), aws(rid=INSTANCE + '\n'), {**aws(), "resourceId": f"arn:aws:ec2:eu-west-1:{ACCOUNT}:instance/{INSTANCE}"}, {**aws(), "region": 'us-east-1"\nprovider "evil" {}'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adoption.classify(value)

    def test_gcp_full_paths_and_identity_conflicts(self):
        row = adoption.classify(gcp())
        self.assertEqual(row["account"], "atlas-prod")
        self.assertEqual(row["zone"], "us-central1-a")
        self.assertEqual(row["region"], "us-central1")
        self.assertEqual(adoption.classify({**gcp(), "resourceId": "//compute.googleapis.com/" + gcp()["resourceId"]})["address"], row["address"])
        with self.assertRaises(ValueError):
            adoption.classify({**gcp(), "account": "other-project"})
        with self.assertRaises(ValueError):
            adoption.classify({**gcp(), "resourceId": "web-prod"})

    def test_nonfinite_cost_and_hcl_expressions(self):
        with self.assertRaises(ValueError):
            adoption.classify({**aws(), "cost": float("inf")})
        self.assertEqual(adoption.hcl_string('${secret}%{if true}"'), '"$${secret}%%{if true}\\\""')


class DiscoveryTests(unittest.TestCase):
    def test_cli_discovery_exports_only_allowlisted_configuration(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            self.assertNotIn("shell", kwargs)
            if "get-caller-identity" in command:
                response = {"Account": ACCOUNT, "Arn": "private"}
            else:
                response = {"Reservations": [{"Instances": [{"InstanceId": INSTANCE, "InstanceType": "t3.medium", "State": {"Name": "running"}, "UserData": "SECRET", "Tags": [{"Value": "SECRET"}], "Placement": {"AvailabilityZone": "us-east-1a"}}]}]}
            return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

        with patch.object(adoption.subprocess, "run", side_effect=run):
            result = adoption.discover_resource(adoption.classify(aws()), "production")
        self.assertEqual(result["instanceType"], "t3.medium")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(calls[0][:3], ["aws", "--profile", "production"])
        self.assertIn("describe-instances", calls[1])
        self.assertNotIn("apply", str(calls))

    def test_wrong_aws_account_never_calls_ec2(self):
        with patch.object(adoption, "run_json", return_value={"Account": "234567890123"}) as run:
            with self.assertRaisesRegex(adoption.CloudError, "different account"):
                adoption.discover_resource(adoption.classify(aws()))
            self.assertEqual(run.call_count, 1)

    def test_gcp_command_uses_explicit_project_and_zone(self):
        with patch.object(adoption, "run_json", return_value={"name": "web-prod", "zone": "https://example/zones/us-central1-a", "machineType": "e2-medium", "metadata": {"secret": "SECRET"}}) as run:
            result = adoption.discover_resource(adoption.classify(gcp()))
        self.assertIn("atlas-prod", run.call_args.args[0])
        self.assertIn("us-central1-a", run.call_args.args[0])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_failures_are_sanitized_and_missing_is_distinct(self):
        for message, missing in [("InvalidInstanceID.NotFound: SECRET", True), ("AccessDenied SECRET", False)]:
            with patch.object(adoption.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", message)):
                with self.assertRaises(adoption.CloudError) as raised:
                    adoption.run_json(["aws", "ec2"])
                self.assertEqual(raised.exception.missing, missing)
                self.assertNotIn("SECRET", str(raised.exception))

    def test_repository_excludes_managed_deleted_and_overlapping_disks(self):
        records = [aws(), aws(VOLUME), gcp(), gcp("disks", "old-disk")]

        def discover(row, *args):
            if row["resourceId"].endswith("old-disk"):
                raise adoption.CloudError("No longer found", True)
            return {"volumeIds": [VOLUME]} if row["resourceType"] == "aws_instance" else {}

        with tempfile.TemporaryDirectory() as temp, patch.object(adoption, "discover_resource", side_effect=discover):
            folder = Path(temp) / "adoption"
            result = adoption.discover({"schemaVersion": 1, "resources": records}, folder, managed=[gcp()])
            self.assertEqual([r["status"] for r in result["resources"]], ["verified", "dependency-review", "already-managed", "not-found"])
            self.assertEqual(len(json.loads((folder / "adoption.json").read_text())["resources"]), 1)
            self.assertIn(INSTANCE, (folder / "imports.tf").read_text())
            self.assertNotIn(VOLUME, (folder / "imports.tf").read_text())
            self.assertIn('allowed_account_ids = ["123456789012"]', (folder / "providers.tf").read_text())
            self.assertIn("*.tfstate*", (folder / ".gitignore").read_text())

    def test_manifest_validation_and_existing_folder_precede_cloud_access(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(adoption, "discover_resource") as cloud:
            for manifest in [{"schemaVersion": 1, "resources": [aws(), aws()]}, {"schemaVersion": 1, "resources": [aws(), {**gcp(), "resourceId": "name-only"}]}, {"schemaVersion": 1, "resources": []}]:
                with self.assertRaises(ValueError):
                    adoption.discover(manifest, Path(temp) / "new")
            with self.assertRaisesRegex(ValueError, "already exists"):
                adoption.discover({"schemaVersion": 1, "resources": [aws()]}, temp)
            cloud.assert_not_called()


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.expected = [adoption.classify(aws()), adoption.classify(gcp())]

    def test_import_only_plan_succeeds_without_exporting_plan_secrets(self):
        value = plan(self.expected)
        value["planned_values"] = {"secret": "SECRET"}
        review = adoption.check_plan(value, self.expected)
        self.assertEqual(review["imports"], 2)
        self.assertEqual(review["infrastructureChanges"], 0)
        self.assertNotIn("SECRET", json.dumps(review))

    def test_every_mutating_action_is_blocked(self):
        for action in [["create"], ["update"], ["delete"], ["delete", "create"], ["create", "delete"], ["read"], []]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                value = plan(self.expected)
                value["resource_changes"][0]["change"]["actions"] = action
                adoption.check_plan(value, self.expected)

    def test_wrong_missing_duplicate_existing_import_and_incomplete_plans_blocked(self):
        variations = []
        value = plan(self.expected); value["resource_changes"][0]["change"]["importing"]["id"] = "other"; variations.append(value)
        value = plan(self.expected); del value["resource_changes"][0]["change"]["importing"]; variations.append(value)
        value = plan(self.expected); value["resource_changes"].pop(); variations.append(value)
        value = plan(self.expected); value["resource_changes"].append(value["resource_changes"][0]); variations.append(value)
        value = plan(self.expected); value["resource_changes"][0]["address"] = "evil.resource"; variations.append(value)
        value = plan(self.expected); value["complete"] = False; variations.append(value)
        value = plan(self.expected); value["deferred_changes"] = [{}]; variations.append(value)
        for value in variations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adoption.check_plan(value, self.expected)

    def test_confirmation_and_changed_plan_prevent_apply(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(adoption, "engine_run") as run:
            folder = Path(temp)
            with self.assertRaisesRegex(ValueError, "confirm-unmanaged"):
                adoption.adopt(folder)
            adoption.write_json(folder / "adoption-plan-review.json", {"planSha256": "old"})
            with patch.object(adoption, "saved_plan", return_value={"planSha256": "new"}):
                with self.assertRaisesRegex(ValueError, "changed after review"):
                    adoption.adopt(folder, confirm=True)
            run.assert_not_called()

    def test_adopt_backs_up_state_and_checks_subsequent_clean_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            review = {"planSha256": "same", "resources": [{"address": self.expected[0]["address"]}]}
            adoption.write_json(folder / "adoption-plan-review.json", review)
            adoption.write_json(folder / "inventory.json", {"resources": [{**self.expected[0], "status": "verified"}]})
            adoption.write_json(folder / "adoption.json", {"resources": self.expected[:1]})
            calls = []

            def run(engine, directory, args, capture=False):
                calls.append(args)
                return 'existing.resource\n' if args == ["state", "list"] else '{"private": "state"}' if args == ["state", "pull"] else json.dumps({"values": {"root_module": {"resources": [{"address": self.expected[0]["address"], "values": {"id": self.expected[0]["importId"]}}]}}}) if args == ["show", "-json"] else None

            with patch.object(adoption, "saved_plan", return_value=review), patch.object(adoption, "engine_run", side_effect=run):
                result = adoption.adopt(folder, confirm=True)
            self.assertEqual(calls[0], ["state", "pull"])
            self.assertIn(["apply", "-input=false", "adoption.tfplan"], calls)
            self.assertEqual(calls[-2:], [["plan", "-input=false", "-detailed-exitcode"], ["show", "-json"]])
            self.assertEqual(result["resources"][0]["status"], "imported")
            backups = list(folder.glob("state-backup-*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)

    def test_failed_post_import_plan_never_marks_resources_imported(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            adoption.write_json(folder / "adoption-plan-review.json", {"planSha256": "same"})
            adoption.write_json(folder / "inventory.json", {"resources": [{"address": "test.resource", "status": "verified"}]})

            def run(engine, directory, args, capture=False):
                if "-detailed-exitcode" in args:
                    raise ValueError("Plan has changes")
                return ""

            with patch.object(adoption, "saved_plan", return_value={"planSha256": "same", "resources": [{"address": "test.resource"}]}), patch.object(adoption, "engine_run", side_effect=run):
                with self.assertRaises(ValueError):
                    adoption.adopt(folder, confirm=True)
            self.assertEqual(adoption.read_json(folder / "inventory.json")["resources"][0]["status"], "imported-needs-review")
            adoption.write_json(folder / "adoption.json", {"resources": [{"address": "test.resource", "importId": "expected"}]})
            state = json.dumps({"values": {"root_module": {"resources": [{"address": "test.resource", "values": {"id": "expected"}}]}}})
            with patch.object(adoption, "engine_run", return_value=state):
                result = adoption.verify_baseline(folder)
            self.assertEqual(result["resources"][0]["status"], "imported")

    def test_partial_apply_failure_is_recorded_and_cannot_look_unmanaged(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            review = {"planSha256": "same", "resources": [{"address": self.expected[0]["address"]}]}
            adoption.write_json(folder / "adoption-plan-review.json", review)
            adoption.write_json(folder / "inventory.json", {"resources": [{**self.expected[0], "status": "verified"}]})

            def run(engine, directory, args, capture=False):
                if "apply" in args:
                    raise ValueError("Interrupted")
                return ""

            with patch.object(adoption, "saved_plan", return_value=review), patch.object(adoption, "engine_run", side_effect=run):
                with self.assertRaisesRegex(ValueError, "partial imports"):
                    adoption.adopt(folder, confirm=True)
            self.assertEqual(adoption.read_json(folder / "inventory.json")["resources"][0]["status"], "import-state-unknown")

    def test_new_workspace_needs_no_backup_and_backend_failure_blocks_adoption(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(adoption, "engine_run", return_value="") as run:
                self.assertIsNone(adoption.backup_state(folder))
                run.assert_called_once_with("terraform", folder, ["state", "pull"], capture=True)
            review = {"planSha256": "same", "resources": []}
            adoption.write_json(folder / "adoption-plan-review.json", review)
            with patch.object(adoption, "saved_plan", return_value=review), patch.object(adoption, "engine_run", side_effect=ValueError("Backend denied")) as run:
                with self.assertRaisesRegex(ValueError, "Backend denied"):
                    adoption.adopt(folder, confirm=True)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[2], ["state", "pull"])

    def test_clean_plan_alone_cannot_verify_missing_or_wrong_state_identity(self):
        for actual in [[], [{"address": self.expected[0]["address"], "values": {"id": "wrong"}}]]:
            with self.assertRaises(ValueError):
                adoption.check_baseline_state({"values": {"root_module": {"resources": actual}}}, self.expected)


class BillingIntegrationTests(unittest.TestCase):
    def test_python_keeps_account_region_resource_evidence_and_unidentified_cost(self):
        content = f'lineItem/UnblendedCost,lineItem/ResourceId,lineItem/UsageAccountId,product/regionCode,lineItem/ProductCode,lineItem/CurrencyCode\n10.125,{INSTANCE},{ACCOUNT},us-east-1,AmazonEC2,USD\n2.125,{INSTANCE},234567890123,us-east-1,AmazonEC2,USD\n-1,,{ACCOUNT},us-east-1,AmazonEC2,USD\n'
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "billing.csv"; file.write_text(content)
            state = cur_to_json.load_cur(file, cur_to_json.DEFAULT_NAMING_CONVENTION)
            summary = cur_to_json.build_summary(state)
            self.assertEqual(len(summary["resourceInventory"]), 2)
            self.assertEqual(summary["resourceInventory"][0]["resourceId"], INSTANCE)
            self.assertEqual(summary["resourceInventory"][0]["region"], "us-east-1")
            self.assertEqual(summary["resourceInventorySummary"], {"unidentifiedRows": 1, "unidentifiedCost": -1.0})
            merged = cur_to_json.merge_cur_states([state, state])
            self.assertEqual(len(merged["resourceInventory"]), 4)
            self.assertEqual(merged["resourceInventorySummary"]["unidentifiedRows"], 2)

    def test_flat_gcp_export_keeps_full_identity(self):
        content = 'cost,service.description,resource.global_name,project.id,location.zone,currency\n3,Compute Engine,//compute.googleapis.com/projects/atlas-prod/zones/us-central1-a/instances/web-prod,atlas-prod,us-central1-a,USD\n'
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "gcp.csv"; file.write_text(content)
            summary = cur_to_json.build_summary(cur_to_json.load_cur(file, cur_to_json.DEFAULT_NAMING_CONVENTION))
            self.assertEqual(adoption.classify(summary["resourceInventory"][0])["resourceType"], "google_compute_instance")


if __name__ == "__main__":
    unittest.main()
