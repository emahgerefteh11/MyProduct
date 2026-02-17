import json
import pathlib
import os

# Templates for Terraform resources
TF_HEADER = """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

provider "azurerm" {
  features {}
}

provider "google" {
  project = "my-project-id"
  region  = "us-central1"
}
"""

AWS_COMPUTE_TEMPLATE = """
resource "aws_instance" "app_server_{clean_name}" {{
  ami           = "ami-0c55b159cbfafe1f0" # Amazon Linux 2
  instance_type = "t3.micro"

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
    Product     = "{product}"
    Platform    = "{platform}"
  }}
}}
"""

AZURE_COMPUTE_TEMPLATE = """
resource "azurerm_linux_virtual_machine" "vm_{clean_name}" {{
  name                = "{clean_name}-vm"
  resource_group_name = "rg-{product}-{env}"
  location            = "East US"
  size                = "Standard_F2"
  admin_username      = "adminuser"
  network_interface_ids = []

  admin_ssh_key {{
    username   = "adminuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }}

  os_disk {{
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }}

  source_image_reference {{
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }}

  tags = {{
    Environment = "{env}"
    Product     = "{product}"
    Platform    = "{platform}"
  }}
}}
"""

GCP_COMPUTE_TEMPLATE = """
resource "google_compute_instance" "vm_{clean_name}" {{
  name         = "{clean_name}-vm"
  machine_type = "e2-medium"
  zone         = "us-central1-a"

  boot_disk {{
    initialize_params {{
      image = "debian-cloud/debian-11"
    }}
  }}

  network_interface {{
    network = "default"
  }}

  labels = {{
    environment = "{env}"
    product     = "{product}"
    platform    = "{platform}"
  }}
}}
"""

AWS_S3_TEMPLATE = """
resource "aws_s3_bucket" "bucket_{clean_name}" {{
  bucket = "{clean_name}-{env}"

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
    Product     = "{product}"
    Platform    = "{platform}"
  }}
}}
"""

def clean_name(name):
    return "".join(c if c.isalnum() else "_" for c in name).lower()

def generate_terraform(report_path):
    with open(report_path, 'r') as f:
        data = json.load(f)

    base_output_dir = pathlib.Path("terraform_generated")
    if not base_output_dir.exists():
        base_output_dir.mkdir()

    platforms = data.get("platforms", [])

    for platform_data in platforms:
        platform_name = platform_data.get("platform", "unknown")
        
        # We need to determine the primary provider for this platform based on the data
        # For now, we'll check the "providers" list in the main report or infer from services.
        # But report_data.json separates platforms and providers.
        # We'll default to AWS if not clear, or mixed.
        
        # Actually, let's look at the products within the platform
        products = platform_data.get("products", [])

        for product_data in products:
            product_name = product_data.get("product", "unknown")
            environments = product_data.get("environments", [])

            # If no explicit environments list, create a default one
            if not environments:
                environments = [{"environment": "default"}]

            for env_data in environments:
                env_name = env_data.get("environment", "default")
                
                output_dir = base_output_dir / clean_name(platform_name) / clean_name(product_name) / clean_name(env_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                main_tf_content = [TF_HEADER]
                
                # Check services used in this product/env context
                # The report structure might nest services under products, but environment breakdown
                # is sometimes separate. We'll use the product-level services as a baseline
                # and assume they apply to all environments for this vending machine prototype.
                
                services = product_data.get("services", [])
                
                for service in services:
                    svc_name = service.get("service", "")
                    
                    # Heuristic: Determine provider from the platform or service name
                    # In a real scenario, we'd track provider per resource.
                    # For this prototype, we'll generate resources for ALL providers
                    # if the service type matches (e.g. Compute/VM -> AWS EC2 + Azure VM).
                    # A better approach is to check the 'provider' field if we propagated it.
                    
                    if svc_name == "Compute/VM":
                        main_tf_content.append(AWS_COMPUTE_TEMPLATE.format(
                            clean_name=clean_name(product_name),
                            name=product_name,
                            env=env_name,
                            product=product_name,
                            platform=platform_name
                        ))
                        # Uncomment to generate Azure/GCP variants if we detect them
                        # main_tf_content.append(AZURE_COMPUTE_TEMPLATE.format(...))

                    elif svc_name == "Object Storage":
                        main_tf_content.append(AWS_S3_TEMPLATE.format(
                            clean_name=clean_name(product_name),
                            name=product_name,
                            env=env_name,
                            product=product_name,
                            platform=platform_name
                        ))

                with open(output_dir / "main.tf", "w") as tf_file:
                    tf_file.write("\n".join(main_tf_content))
                
                print(f"Generated {output_dir}/main.tf")

if __name__ == "__main__":
    import sys
    report_file = sys.argv[1] if len(sys.argv) > 1 else "myproduct/report_data.json"
    generate_terraform(report_file)
