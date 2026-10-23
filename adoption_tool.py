#!/usr/bin/env python3
"""Discover billed resources and prepare reviewed, import-only Terraform adoption.

Python standard library only. Cloud commands use the operator's local AWS/gcloud
login. Raw cloud responses and Terraform plan/state JSON are never exported to
the dashboard. No cloud mutations occur during discovery or planning.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MAX_RESOURCES = 200
AWS_REGION = r"[a-z]{2}(?:-[a-z]+)+-\d+"
PROJECT = r"(?:[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{6,20})"
ZONE = r"[a-z]+-[a-z]+\d+-[a-z]"
NAME = r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
SUPPORTED = {"aws_instance", "aws_ebs_volume", "google_compute_instance", "google_compute_disk"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    if Path(path).stat().st_size > 10 * 1024 * 1024:
        raise ValueError("JSON input must be at most 10 MiB.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def text(record, key):
    value = record.get(key, "")
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {key}: use a plain resource identifier.")
    return value.strip()


def classify(record):
    """Validate identity independently of the browser; never guess from a name."""
    item = {k: text(record, k) for k in ("provider", "account", "region", "zone", "resourceId")}
    rid, provider = item["resourceId"], item["provider"]
    resource_type, import_id = "", ""
    if provider == "aws":
        arn = re.fullmatch(r"arn:(aws|aws-us-gov|aws-cn):ec2:(" + AWS_REGION + r"):(\d{12}):(instance|volume)/((?:i|vol)-[0-9a-f]{8}(?:[0-9a-f]{9})?)", rid)
        if arn:
            _, region, account, kind, rid = arn.groups()
            if (kind == "instance") != rid.startswith("i-"):
                raise ValueError("AWS ARN resource type conflicts with its ID.")
            for key, value in (("account", account), ("region", region)):
                if item[key] and item[key] != value:
                    raise ValueError(f"AWS ARN conflicts with the supplied {key}.")
                item[key] = value
        if re.fullmatch(r"i-[0-9a-f]{8}(?:[0-9a-f]{9})?", rid):
            resource_type = "aws_instance"
        elif re.fullmatch(r"vol-[0-9a-f]{8}(?:[0-9a-f]{9})?", rid):
            resource_type = "aws_ebs_volume"
        else:
            raise ValueError("Unsupported AWS resource. Initial support covers EC2 instances and EBS volumes with real resource IDs.")
        if not re.fullmatch(r"\d{12}", item["account"]) or not re.fullmatch(AWS_REGION, item["region"]):
            raise ValueError("AWS discovery requires the 12-digit usage account ID and region code.")
        import_id = rid
    elif provider == "gcp":
        rid = re.sub(r"^(?://compute\.googleapis\.com/|https://www\.googleapis\.com/compute/v1/)", "", rid)
        match = re.fullmatch(r"projects/(" + PROJECT + r")/zones/(" + ZONE + r")/(instances|disks)/(" + NAME + r")", rid)
        if not match:
            raise ValueError("GCP discovery requires a full projects/PROJECT/zones/ZONE/instances/NAME or disks/NAME resource path.")
        project, zone, kind, name = match.groups()
        for key, value in (("account", project), ("zone", zone)):
            if item[key] and item[key] != value:
                raise ValueError(f"GCP resource path conflicts with the supplied {key}.")
            item[key] = value
        region = zone.rsplit("-", 1)[0]
        if item["region"] and item["region"] != region:
            raise ValueError("GCP resource path conflicts with the supplied region.")
        item["region"] = region
        resource_type = "google_compute_instance" if kind == "instances" else "google_compute_disk"
        import_id = f"projects/{project}/zones/{zone}/{kind}/{name}"
    else:
        raise ValueError("Initial adoption support covers AWS and GCP. This provider remains unresolved.")
    item.update(resourceType=resource_type, importId=import_id)
    identity = "|".join(item[k] for k in ("provider", "account", "region", "importId"))
    item["address"] = resource_type + ".atlas_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    # Carry only explicit billing evidence. Discard arbitrary uploaded attributes.
    cost = record.get("cost", 0)
    if not isinstance(cost, (float, int)) or isinstance(cost, bool) or not Decimal(str(cost)).is_finite():
        raise ValueError("Resource cost must be a finite number.")
    item["cost"] = cost
    hashes = record.get("sourceHashes", [])
    item["sourceHashes"] = sorted({h for h in hashes if isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)})
    return item


def inventory_rows(rows, source_hash):
    """Keep billing identity independent of naming conventions and display limits."""
    groups = {}
    unidentified_cost, unidentified_rows = Decimal(0), 0
    for row in rows:
        rid = row["resourceId"]
        if not rid:
            unidentified_cost += row["cost"]
            unidentified_rows += 1
            continue
        key = tuple(row.get(k, "") for k in ("provider", "account", "region", "zone", "resourceId"))
        if key not in groups:
            groups[key] = dict(zip(("provider", "account", "region", "zone", "resourceId"), key), cost=Decimal(0), rows=0, services=set(), sourceHashes=[source_hash])
        item = groups[key]
        item["cost"] += row["cost"]
        item["rows"] += 1
        item["services"].add(row["service"])
    return [{**v, "cost": float(round(v["cost"], 2)), "services": sorted(v["services"])} for v in groups.values()], {"unidentifiedCost": float(round(unidentified_cost, 2)), "unidentifiedRows": unidentified_rows}


class CloudError(ValueError):
    def __init__(self, message, missing=False):
        super().__init__(message)
        self.missing = missing


def run_json(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False, env={**os.environ, "AWS_PAGER": "", "CLOUDSDK_CORE_DISABLE_PROMPTS": "1"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CloudError(f"Could not run {command[0]}. Install the CLI, check login and retry.") from exc
    if result.returncode:
        # Error text may include private data; only expose a bounded diagnosis.
        missing = bool(re.search(r"Invalid(?:InstanceID|Volume)\.NotFound|NOT_FOUND|was not found|does not exist", result.stderr))
        raise CloudError("Resource was not found in the selected scope." if missing else f"{command[0]} read failed. Check login, account/project, region and read permissions.", missing)
    try:
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, TypeError) as exc:
        raise CloudError(f"{command[0]} returned an invalid JSON response.") from exc


def discover_resource(item, aws_profile="", identity_cache=None):
    identity_cache = identity_cache if identity_cache is not None else {}
    rid, kind = item["importId"], item["resourceType"]
    if item["provider"] == "aws":
        prefix = ["aws"] + (["--profile", aws_profile] if aws_profile else [])
        if "aws" not in identity_cache:
            identity_cache["aws"] = run_json(prefix + ["sts", "get-caller-identity", "--output", "json", "--no-cli-pager"]).get("Account")
        if identity_cache["aws"] != item["account"]:
            raise CloudError("AWS login belongs to a different account. Choose the matching AWS profile.")
        if kind == "aws_instance":
            data = run_json(prefix + ["ec2", "describe-instances", "--instance-ids", rid, "--region", item["region"], "--output", "json", "--no-cli-pager"])
            resources = [r for reservation in data.get("Reservations", []) for r in reservation.get("Instances", []) if r.get("InstanceId") == rid]
            if len(resources) != 1 or resources[0].get("State", {}).get("Name") == "terminated":
                raise CloudError("Instance no longer exists in the selected scope.", True)
            r = resources[0]
            return {"instanceType": r.get("InstanceType"), "imageId": r.get("ImageId"), "subnetId": r.get("SubnetId"), "vpcId": r.get("VpcId"),
                    "zone": r.get("Placement", {}).get("AvailabilityZone"), "securityGroupIds": [g.get("GroupId") for g in r.get("SecurityGroups", [])],
                    "volumeIds": [d.get("Ebs", {}).get("VolumeId") for d in r.get("BlockDeviceMappings", [])]}
        data = run_json(prefix + ["ec2", "describe-volumes", "--volume-ids", rid, "--region", item["region"], "--output", "json", "--no-cli-pager"])
        resources = [r for r in data.get("Volumes", []) if r.get("VolumeId") == rid]
        if len(resources) != 1:
            raise CloudError("Volume no longer exists in the selected scope.", True)
        r = resources[0]
        return {"sizeGiB": r.get("Size"), "volumeType": r.get("VolumeType"), "zone": r.get("AvailabilityZone"), "encrypted": r.get("Encrypted"),
                "attachedInstanceIds": [a.get("InstanceId") for a in r.get("Attachments", [])]}
    plural = "instances" if kind == "google_compute_instance" else "disks"
    name = rid.split("/")[-1]
    r = run_json(["gcloud", "compute", plural, "describe", name, "--project", item["account"], "--zone", item["zone"], "--format=json", "--quiet"])
    if r.get("name") != name or r.get("zone", "").split("/")[-1] != item["zone"]:
        raise CloudError("GCP discovery returned a different resource identity.")
    if kind == "google_compute_instance":
        return {"machineType": r.get("machineType"), "status": r.get("status"), "diskIds": [d.get("source") for d in r.get("disks", [])],
                "networkIds": [n.get("network") for n in r.get("networkInterfaces", [])], "subnetworkIds": [n.get("subnetwork") for n in r.get("networkInterfaces", [])]}
    return {"sizeGiB": r.get("sizeGb"), "diskType": r.get("type"), "status": r.get("status"), "attachedInstanceIds": r.get("users", [])}


def hcl_string(value):
    return json.dumps(value).replace("${", "$${").replace("%{", "%%{")


def alias(item):
    return "scope_" + hashlib.sha256((item["provider"] + "|" + item["account"] + "|" + item["region"]).encode()).hexdigest()[:12]


def repository_files(resources, aws_profile=""):
    providers = sorted({r["provider"] for r in resources})
    required = []
    if "aws" in providers:
        required.append('    aws = { source = "hashicorp/aws", version = ">= 5.0, < 7.0" }')
    if "gcp" in providers:
        required.append('    google = { source = "hashicorp/google", version = ">= 6.0, < 8.0" }')
    config = 'terraform {\n  required_version = ">= 1.5, < 2.0"\n  required_providers {\n' + "\n".join(required) + '\n  }\n}\n'
    seen = set()
    for r in resources:
        key = alias(r)
        if key in seen:
            continue
        seen.add(key)
        provider = "aws" if r["provider"] == "aws" else "google"
        config += f'\nprovider "{provider}" {{\n  alias = {hcl_string(key)}\n'
        if provider == "aws":
            config += f'  region = {hcl_string(r["region"])}\n  allowed_account_ids = [{hcl_string(r["account"])}]\n'
            if aws_profile:
                config += f'  profile = {hcl_string(aws_profile)}\n'
        else:
            config += f'  project = {hcl_string(r["account"])}\n  region = {hcl_string(r["region"])}\n'
        config += "}\n"
    imports = "# Configuration will be generated from the live provider by the plan command.\n"
    for r in resources:
        provider = "aws" if r["provider"] == "aws" else "google"
        imports += f'\nimport {{\n  to = {r["address"]}\n  id = {hcl_string(r["importId"])}\n  provider = {provider}.{alias(r)}\n}}\n'
    return {"providers.tf": config, "imports.tf": imports}


REPOSITORY_README = """# Adopt existing resources

Discovery verifies current presence and exports selected configuration evidence.
Terraform/OpenTofu reads the full provider configuration during `plan` and writes
`generated.tf`. Dependencies are preserved as existing IDs, not automatically
adopted. Review generated configuration, including disks, networking, identity,
default values, sensitive fields, and any settings the provider cannot import.
The provider may require configuration corrections before a clean import plan.
No optimization or module refactoring belongs in the initial adoption batch.

Run from the directory containing adoption_tool.py:

    terraform -chdir=adoption init
    python adoption_tool.py plan --directory adoption

Use `--engine tofu` for OpenTofu. Configure AWS credentials (or the same named
profile used for discovery); for GCP configure Application Default Credentials
with permission to read the selected project. gcloud CLI login and ADC are
separate. The account/project in providers.tf is explicit.

Review generated.tf and adoption-plan-review.json. The plan command fails unless
every expected resource is imported with no infrastructure changes. After edits,
run plan again. Once reviewed, import with:

    python adoption_tool.py adopt --directory adoption --confirm-unmanaged

This rechecks the saved plan, backs up any current state, imports that exact plan,
and requires a clean subsequent plan. If import succeeds but the subsequent plan
fails, inventory.json records imported-needs-review. Resolve configuration
differences without changing infrastructure and run:

    python adoption_tool.py verify --directory adoption

If apply fails or is interrupted, inventory records import-state-unknown because
some imports may already exist. Inspect state and recover through your normal
Terraform workflow before retrying. Verification checks both a clean plan and
the original resource IDs in state.

Only resources verified as unmanaged by
your organization belong here. Use --managed-ids during discovery to exclude a
known inventory; discovery cannot detect all external Terraform states.

State, saved plans, generated.tf and backups can contain secrets. They are
ignored by Git. Review/redact generated.tf before intentionally tracking it.
Configure your organization's protected remote backend with locking, encryption
and versioning before shared use; no backend or cloud resources are created by
this tool. Keep .terraform.lock.hcl in version control. Commit reviewed config,
then add your normal CI plan/review pipeline before future changes.

inventory.json is the allowlisted discovery summary you can reopen in the Atlas
adoption screen. Do not upload raw plan/state JSON to the dashboard. A verified
import manages existing resources; it does not verify application behavior or
discover every dependency or resource absent from billing.
"""


def discover(manifest, output, aws_profile="", managed=None):
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("resources"), list):
        raise ValueError("Expected an Atlas adoption manifest with schemaVersion 1 and resources.")
    if not 1 <= len(manifest["resources"]) <= MAX_RESOURCES:
        raise ValueError(f"Choose between 1 and {MAX_RESOURCES} resources per adoption batch.")
    if aws_profile and (aws_profile.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@+=,-]{1,128}", aws_profile)):
        raise ValueError("Choose a valid local AWS profile name.")
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory already exists. Choose a new directory to preserve existing configuration and state.")
    # Validate the entire input before making any cloud calls.
    resources = [classify(r) for r in manifest["resources"]]
    if len({r["address"] for r in resources}) != len(resources):
        raise ValueError("The batch contains duplicate cloud identities. Select each resource only once.")
    managed_addresses = {classify(r)["address"] for r in (managed or [])}
    cache, results = {}, []
    for r in resources:
        item = {**r, "checkedAt": stamp()}
        if r["address"] in managed_addresses:
            item.update(status="already-managed", message="Excluded by your supplied managed-resource inventory.")
        else:
            try:
                item.update(configuration=discover_resource(r, aws_profile, cache), status="verified", message="Live resource found; configuration generation and import review are next.")
            except CloudError as exc:
                item.update(status="not-found" if exc.missing else "error", message=str(exc))
        results.append(item)
    # An instance's generated inline disk blocks must not compete with separate
    # disk ownership in the same initial adoption batch.
    attached = set()
    for r in results:
        if r["status"] == "verified" and r["resourceType"] in {"aws_instance", "google_compute_instance"}:
            for disk in r["configuration"].get("volumeIds", []) + r["configuration"].get("diskIds", []):
                if disk:
                    attached.add((r["provider"], r["account"], r["region"], disk.removeprefix("https://www.googleapis.com/compute/v1/")))
    for r in results:
        if (r["provider"], r["account"], r["region"], r["importId"]) in attached and r["status"] == "verified":
            r.update(status="dependency-review", message="Disk is also represented by a selected instance. Adopt the instance first and review disk ownership before a separate batch.")
    ready = [r for r in results if r["status"] == "verified"]
    output.mkdir(parents=True, exist_ok=False)
    inventory = {"schemaVersion": 1, "kind": "atlas-adoption-inventory", "generatedAt": stamp(), "resources": results}
    write_json(output / "inventory.json", inventory)
    write_json(output / "adoption.json", {"schemaVersion": 1, "resources": ready})
    if ready:
        for name, content in repository_files(ready, aws_profile).items():
            (output / name).write_text(content, encoding="utf-8")
    (output / "README.md").write_text(REPOSITORY_README, encoding="utf-8")
    (output / ".gitignore").write_text(".terraform/\n*.tfstate*\n*.tfplan\n*.tfplan.*\n*.tfvars\n*.tfvars.json\ngenerated.tf\nstate-backup-*.json\ncrash.log\n", encoding="utf-8")
    return inventory


def check_plan(plan, expected):
    """Reject all infrastructure changes, unknown imports and incomplete plans."""
    if not isinstance(plan, dict) or plan.get("errored") or plan.get("complete") is False or plan.get("deferred_changes"):
        raise ValueError("Plan is incomplete, errored, or contains deferred changes.")
    if not expected:
        raise ValueError("There are no verified resources to import.")
    wanted = {r["address"]: r["importId"] for r in expected}
    if len(wanted) != len(expected):
        raise ValueError("Duplicate adoption addresses.")
    found = set()
    for change in plan.get("resource_changes", []):
        address = change.get("address")
        detail = change.get("change", {})
        if address not in wanted or address in found or detail.get("actions") != ["no-op"]:
            raise ValueError("Adoption blocked: plan contains an unexpected resource or a create, update, replacement, delete, or read action.")
        importing = detail.get("importing") or {}
        if importing.get("id") != wanted[address] or importing.get("unknown"):
            raise ValueError("Adoption blocked: a resource is already managed, has an unknown import ID, or imports a different ID.")
        found.add(address)
    if found != set(wanted):
        raise ValueError("Adoption blocked: the plan does not import every verified resource.")
    if any(d.get("change", {}).get("actions") != ["no-op"] for d in plan.get("resource_drift", [])):
        raise ValueError("Adoption blocked: resolve reported resource drift first.")
    return {"schemaVersion": 1, "kind": "atlas-adoption-plan-review", "checkedAt": stamp(), "imports": len(found), "infrastructureChanges": 0,
            "resources": [{"address": r["address"], "importId": r["importId"]} for r in expected]}


def engine_run(engine, directory, args, capture=False):
    try:
        result = subprocess.run([engine, f"-chdir={directory}"] + args, capture_output=capture, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not run {engine}. Install it, initialize the adoption directory and retry.") from exc
    if result.returncode:
        raise ValueError(f"{engine} command failed (exit {result.returncode}). Review configuration and credentials locally, then retry.")
    return result.stdout if capture else None


def saved_plan(directory, engine):
    expected = read_json(directory / "adoption.json").get("resources", [])
    # Recompute addresses/import IDs so metadata edits cannot change the target silently.
    canonical = [classify(r) for r in expected]
    if any(a["address"] != b.get("address") or a["importId"] != b.get("importId") for a, b in zip(canonical, expected)):
        raise ValueError("Adoption metadata does not match its resource identities.")
    plan = json.loads(engine_run(engine, directory, ["show", "-json", "adoption.tfplan"], capture=True))
    review = check_plan(plan, canonical)
    review["planSha256"] = hashlib.sha256((directory / "adoption.tfplan").read_bytes()).hexdigest()
    return review


def plan_adoption(directory, engine="terraform"):
    directory = Path(directory).resolve()
    receipt = directory / "adoption-plan-review.json"
    receipt.unlink(missing_ok=True)
    if not read_json(directory / "adoption.json").get("resources"):
        raise ValueError("No verified resources. Resolve discovery gaps before planning.")
    args = ["plan", "-input=false", "-out=adoption.tfplan"]
    if not (directory / "generated.tf").exists():
        args.append("-generate-config-out=generated.tf")
    engine_run(engine, directory, args)
    review = saved_plan(directory, engine)
    write_json(receipt, review)
    return review


def backup_state(directory, engine="terraform"):
    # state pull succeeds with no output when no state exists. Unlike state
    # list, it therefore supports a brand-new organization/workspace while
    # still failing closed on backend/authentication/read errors.
    backup = engine_run(engine, directory, ["state", "pull"], capture=True)
    if not backup.strip():
        return None
    json.loads(backup)  # Never keep a non-JSON response as a state backup.
    backup_path = directory / ("state-backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    with os.fdopen(os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(backup)
    return backup_path


def adopt(directory, engine="terraform", confirm=False):
    if not confirm:
        raise ValueError("Confirm these resources are not owned by another IaC state using --confirm-unmanaged after your review.")
    directory = Path(directory).resolve()
    receipt = read_json(directory / "adoption-plan-review.json")
    review = saved_plan(directory, engine)
    if receipt.get("planSha256") != review["planSha256"]:
        raise ValueError("The saved plan changed after review. Run plan and review again.")
    backup_state(directory, engine)
    inventory = read_json(directory / "inventory.json")
    imported = {r["address"] for r in review["resources"]}
    for r in inventory["resources"]:
        if r["address"] in imported:
            r.update(status="import-state-unknown", message="Import step started; inspect state if this process does not complete.", checkedAt=stamp())
    write_json(directory / "inventory.json", inventory)
    try:
        engine_run(engine, directory, ["apply", "-input=false", "adoption.tfplan"])
    except (ValueError, KeyboardInterrupt) as exc:
        # An interrupted apply can already have imported some resources. Never
        # present that batch as fresh/unmanaged or automatically retry it.
        for r in inventory["resources"]:
            if r["address"] in imported:
                r.update(status="import-state-unknown", message="Import failed or was interrupted; state may contain partial imports. Inspect state before retrying this batch.", checkedAt=stamp())
        write_json(directory / "inventory.json", inventory)
        raise ValueError("Import failed; partial imports may exist. Inspect your Terraform state and inventory.json before retrying. After recovery, run verify to check the baseline.") from exc
    for r in inventory["resources"]:
        if r["address"] in imported:
            r.update(status="imported-needs-review", message="Import completed. A clean subsequent plan is required to verify the baseline.", checkedAt=stamp())
    write_json(directory / "inventory.json", inventory)
    return verify_baseline(directory, engine)


def verify_baseline(directory, engine="terraform"):
    directory = Path(directory).resolve()
    inventory = read_json(directory / "inventory.json")
    if not any(r.get("status") in {"imported", "imported-needs-review", "import-state-unknown"} for r in inventory.get("resources", [])):
        raise ValueError("No completed imports are recorded. Run the guarded adoption step first.")
    # Exit code 2 (changes) is deliberately a failure for baseline verification.
    try:
        engine_run(engine, directory, ["plan", "-input=false", "-detailed-exitcode"])
    except ValueError as exc:
        raise ValueError("Baseline verification failed. Review the local plan, state and configuration; run verify again after resolving differences. The inventory retains the unresolved import status.") from exc
    expected = read_json(directory / "adoption.json")["resources"]
    state = json.loads(engine_run(engine, directory, ["show", "-json"], capture=True))
    check_baseline_state(state, expected)
    for r in inventory["resources"]:
        if r.get("status") in {"imported", "imported-needs-review", "import-state-unknown"}:
            r.update(status="imported", message="Imported and verified by a clean subsequent plan.", checkedAt=stamp())
    write_json(directory / "inventory.json", inventory)
    return inventory


def check_baseline_state(state, expected):
    resources = state.get("values", {}).get("root_module", {}).get("resources", [])
    actual = {r.get("address"): r.get("values", {}).get("id") for r in resources}
    if not expected or any(actual.get(r["address"]) != r["importId"] for r in expected):
        raise ValueError("Baseline state does not contain every expected resource with its original import ID. Inspect state before marking adoption complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover", help="Read current AWS/GCP resource configuration; create an adoption repository.")
    discovery.add_argument("--manifest", required=True, type=Path)
    discovery.add_argument("--output", default="adoption", type=Path)
    discovery.add_argument("--aws-profile", default="")
    discovery.add_argument("--managed-ids", type=Path, help="JSON object with resources in the same identity format; excluded from adoption.")
    for command in ("plan", "adopt", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--directory", default="adoption", type=Path)
        sub.add_argument("--engine", choices=["terraform", "tofu"], default="terraform")
        if command == "adopt":
            sub.add_argument("--confirm-unmanaged", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "discover":
            managed = read_json(args.managed_ids)["resources"] if args.managed_ids else []
            result = discover(read_json(args.manifest), args.output, args.aws_profile, managed)
            ready = sum(r["status"] == "verified" for r in result["resources"])
            print(f"Discovery complete: {ready}/{len(result['resources'])} resources verified. Review {args.output}/inventory.json and README.md.")
            return 0 if ready else 2
        if args.command == "plan":
            review = plan_adoption(args.directory, args.engine)
            print(f"Verified {review['imports']} imports with zero infrastructure changes. Review generated.tf and adoption-plan-review.json before adoption.")
        elif args.command == "adopt":
            adopt(args.directory, args.engine, args.confirm_unmanaged)
            print("Adoption complete; subsequent plan is clean. Reopen inventory.json in Atlas to update coverage.")
        else:
            verify_baseline(args.directory, args.engine)
            print("Imported baseline verified by a clean plan. Reopen inventory.json in Atlas to update coverage.")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Adoption stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
