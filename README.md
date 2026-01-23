This is a personal project I am working on. The intent is to be an all-in-one place for infrastructure: analyzing existing infrastructure, breaking down cost by platform/product, categorizing SKUs per platform/product, providing FinOps views, and eventually producing Terraform to provision future builds. Currently it's AWS-only with mock data.

## Naming conventions
The report can derive product/platform/environment from resource names. If you do not provide a naming convention file, it falls back to the Microsoft CAF pattern: `<resource_type>-<workload>-<environment>-<region>-<instance>`.

Examples:
- `python cur_to_json.py --input MyProduct/mock_aws_cur_iaas_heavy.csv --output MyProduct/report_data.json --naming naming_convention.sample.json`
- `.\run_cur_report.ps1 -CsvPath MyProduct/mock_aws_cur_iaas_heavy.csv -NamingPath .\naming_convention.sample.json`
- `python cur_to_json.py --input jan.csv --input feb.csv --label Jan --label Feb --output MyProduct/report_data.json`
- `.\run_cur_report.ps1 -CsvPath jan.csv,feb.csv -Label Jan,Feb`

Tip: when using `-File` with PowerShell, pass comma-separated lists as a single string (example: `-CsvPath "jan.csv,feb.csv"`).

By default, tag values win when present. To force naming conventions to override tags, add `--naming-mode override` (or `-NamingMode override` for the PowerShell script).

## Service normalization
Service costs are normalized into shared families (ex: AWS EC2 + Azure Virtual Machines -> `Compute/VM`, EBS + Managed Disks -> `Block Storage`) so cross-cloud dashboards aggregate comparable offerings.

Azure VM size breakdowns are derived from `SkuName` (or `MeterName`) when available; unknown sizes fall back to `Unknown size`.

## Release Roadmap (draft)
- **AWS (current):** CUR ingestion, FinOps overview, platform/product filters, resource naming breakdown, anomalies/distributions (EC2 SKUs, EBS/S3 buckets).
- **Upcoming:** 
  - GCP & Azure analysis parity (billing ingestion, SKU mapping, platform/product breakdowns).
  - Terraform buildouts: generate TF skeletons for discovered stacks (networking, compute, storage) per platform/product.
  - Resource naming convention inputs: input resource naming conventions related to products/platforms to produce reports by product/platform.
  - Resource naming convention analysis: determining resource naming conventions as they relate to platforms/products without inputs.

AWS data in this repo is mock and AI-generated.
