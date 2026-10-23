"""Validate all four supported import types against real Terraform providers.

Runs in CI with Terraform installed. No cloud credentials or cloud calls.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adoption_tool import alias, classify, repository_files


def main():
    resources = [
        classify({"provider": "aws", "account": "123456789012", "region": "us-east-1", "resourceId": "i-0123456789abcdef0"}),
        classify({"provider": "aws", "account": "123456789012", "region": "us-east-1", "resourceId": "vol-0123456789abcdef0"}),
        classify({"provider": "gcp", "resourceId": "projects/atlas-prod/zones/us-central1-a/instances/web-prod"}),
        classify({"provider": "gcp", "resourceId": "projects/atlas-prod/zones/us-central1-a/disks/data-prod"}),
    ]
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for name, content in repository_files(resources).items():
            (folder / name).write_text(content)
        stubs = []
        # validate requires resource blocks; normal adoption obtains these from
        # plan -generate-config-out after live provider reads.
        for r in resources:
            provider = "aws" if r["provider"] == "aws" else "google"
            kind, name = r["address"].split(".")
            body = {
                "aws_instance": 'ami = "ami-0123456789abcdef0"\ninstance_type = "t3.medium"',
                "aws_ebs_volume": 'availability_zone = "us-east-1a"\nsize = 10',
                "google_compute_instance": 'name = "web-prod"\nmachine_type = "e2-medium"\nzone = "us-central1-a"\nboot_disk {\n  initialize_params {\n    image = "debian-cloud/debian-12"\n  }\n}\nnetwork_interface { network = "default" }',
                "google_compute_disk": 'name = "data-prod"\nzone = "us-central1-a"\nsize = 10',
            }[kind]
            stubs.append(f'resource "{kind}" "{name}" {{\nprovider = {provider}.{alias(r)}\n{body}\n}}')
        (folder / "validation_resources.tf").write_text("\n".join(stubs))
        for args in [["fmt", "-recursive"], ["init", "-backend=false", "-input=false", "-no-color"], ["validate", "-no-color"]]:
            subprocess.run(["terraform", f"-chdir={folder}"] + args, check=True, timeout=300)


if __name__ == "__main__":
    main()
