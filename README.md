# Cloud Cost Atlas (MyProduct)

[![Multi-Cloud](https://img.shields.io/badge/Multi--Cloud-AWS%20%7C%20Azure%20%7C%20GCP-blue)](https://github.com/emahgerefteh11/MyProduct)
[![FinOps](https://img.shields.io/badge/FinOps-Foundation%20Aligned-green)](https://www.finops.org/)
[![IaC](https://img.shields.io/badge/IaC-Terraform%20Vending-purple)](https://www.terraform.io/)
[![Static Dashboard](https://img.shields.io/badge/Dashboard-HTML5%20%2F%20JS-orange)](dashboard.html)

**Cloud Cost Atlas** is a multi-cloud infrastructure intelligence and FinOps tool. It ingests supported billing CSV formats from **AWS**, **Microsoft Azure**, and **Google Cloud Platform (GCP)**, normalizes services into a unified taxonomy, extracts hardware sizing and naming conventions, generates interactive dashboards, and creates Terraform starter templates based on reported cloud footprints. Generated infrastructure requires review and validation before deployment.

## Report trust and data quality

The report dashboard displays imported costs, not a real-time cloud connection. Its provenance panel shows the actual analysis timestamp, source filenames and SHA-256 fingerprints, row counts, observed usage dates, currency, cost basis, and source-to-report reconciliation. Unallocated product, platform, and environment costs are recorded in the report. Reconciliation checks the imported rows; it does not establish completeness against a cloud invoice.

- **Report mode:** savings, month-over-month changes, and commitment coverage show unavailable states until supporting evidence exists. Billing alone cannot establish idle resources or rightsizing opportunities. Older JSON reports still load, with unknown provenance and currency clearly indicated.
- **Demo mode:** simulated insights remain available for presentations and are explicitly labeled. Missing demo currency is illustrated in USD. Charts and fonts use external CDNs; when the chart library is unavailable, tables and analysis remain usable.
- **Bundled sample:** the checked-in `report_data.json` is explicitly marked as sample data and opens in demo mode even through the report page. Running the analysis CLI replaces it with a report marked `sampleData: false`; reports generated from your own mock inputs should still be treated as samples.
- **Currency:** source currencies are retained. If a file lacks currency and you know its currency, pass `--currency USD` (or another three-letter code). This only fills missing values; it never performs FX conversion. Mixed currencies, or known currencies combined with unspecified currencies, are rejected. Reports with entirely unspecified currencies show numeric amounts without currency symbols and include a warning.
- **Validation:** invalid or non-finite amounts, empty inputs, identical file contents supplied twice, and mixed billed/effective cost bases are rejected before replacing the output report. Duplicate detection covers whole files, not overlapping rows across different exports.
- **Attribution:** native AWS `resourceTags/user:*` columns, Azure JSON tags, and FOCUS/GCP JSON tags or key/value label arrays are supported. Cloud-generated IDs are not interpreted as product names. FOCUS provider values are read from rows and normalized; this is partial FOCUS ingestion, not a conformance certification.

For example, the included sample files omit currency, so their intended USD denomination must be explicit:

```bash
python cur_to_json.py \
  --input mock_aws_cur_iaas_heavy_named20251122.csv \
  --input mock_gcp_billing.csv \
  --naming naming_convention.sample.json \
  --currency USD --output report_data.json
```

Real period comparison, guided billing upload, verified savings, and infrastructure-change workflows remain upcoming iterations. The current multi-file summaries do not establish comparable monthly periods.

New reports preserve provider attribution per platform/product. The IaC builders use that evidence instead of inferring a cloud from the team name. The browser builder blocks ambiguous providers; the CLI requires an explicit `--provider` choice for those products. Legacy reports retain heuristic provider selection. Templates remain unvalidated starter configurations.

### Regression checks

The analysis engine and its tests use the Python standard library. Dashboard behavior checks use Node.js and a minimal DOM harness; they do not require a browser or CDN access.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/dashboard.test.cjs
python generate_dashboard.py
```

GitHub Actions runs these checks and verifies that the generated demo matches `dashboard.html` and `report_data.json`.

---

## Key Capabilities

```
                       ┌─────────────────────────────────────────────────────────┐
                       │               Multi-Cloud Billing Ingestion             │
                       │   AWS CUR  │  Azure Cost Management  │  GCP Standard   │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │          Service Normalization & Naming Parser          │
                       │   • 8 Unified Functional Families (Compute, Storage...) │
                       │   • Microsoft CAF Naming Convention Engine              │
                       │   • Tag & Dimension Reconciliation (Fallback/Override)  │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │            Hardware & FinOps Metric Engine              │
                       │   • vCPU, RAM GiB, and Storage GB-Mo Sizing             │
                       │   • HHI Concentration & Shared Stack Signatures         │
                       │   • Multi-Period Sequential Timeline Aggregation        │
                       └─────────────────────┬──────────────────┬────────────────┘
                                             │                  │
                      ┌──────────────────────▼───┐          ┌───▼──────────────────────┐
                      │    Visual Dashboards     │          │    Terraform Vending     │
                      │  • Live (dashboard.html) │          │  • Modular Stacks        │
                      │  • Demo (Self-contained) │          │  • SKU-Mapped Resources  │
                      └──────────────────────────┘          └──────────────────────────┘
```

1. **Unified Multi-Cloud Ingestion:** Ingests native billing exports across AWS CUR, Azure Cost Management, and GCP Standard Billing without requiring external SaaS dependencies.
2. **Cross-Cloud Service Normalization:** Maps vendor-specific service lines (e.g., EC2 vs Azure VMs vs Compute Engine, EBS vs Managed Disks vs Persistent Disks) into 8 standardized functional categories.
3. **Enterprise Naming Convention Engine:** Derives product, platform, environment, and region from resource identifiers using the **Microsoft Azure Cloud Adoption Framework (CAF)** standard or custom JSON schemas.
4. **Hardware Capacity Introspection:** Translates cloud billing SKUs into tangible compute specs (vCPU count, RAM GiB) and storage capacity (GB-Mo) for capacity planning.
5. **Dual-Mode Interactive Dashboard:** 
   - **Live Mode (`dashboard.html`):** Dynamically fetches `report_data.json` generated by the analysis engine.
   - **Demo Mode (`dashboard_demo.html`):** HTML file with embedded data and labeled simulations, ready for presentations and GitHub Pages deployment. Charts require access to the chart-library CDN.
6. **Intelligent Terraform Vending (`terraform_generator.py`):** Automatically scaffolds modular Terraform configurations (`main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars`) parameterized with discovered instance types, storage allocations, and tags.

---

## Service Normalization Taxonomy

Disparate CSP offerings are normalized into 8 shared categories so cross-cloud dashboards aggregate comparable workloads:

| Unified Family | AWS Equivalent | Azure Equivalent | GCP Equivalent |
| :--- | :--- | :--- | :--- |
| **Compute/VM** | Amazon EC2 | Virtual Machines, VMSS | Compute Engine |
| **Block Storage** | Amazon EBS | Managed Disks | Persistent Disks |
| **Object Storage** | Amazon S3 | Blob Storage, Storage Accounts | Cloud Storage (GCS) |
| **Database** | Amazon RDS, DynamoDB | Azure SQL, Cosmos DB | Cloud SQL, Spanner |
| **Cache** | Amazon ElastiCache, MemoryDB | Azure Cache for Redis | Cloud Memorystore |
| **Container** | Amazon EKS, ECS, ECR | Azure Kubernetes Service (AKS) | Google Kubernetes Engine (GKE) |
| **Networking** | Amazon VPC, ELB, NAT Gateway | Virtual Network, Load Balancer | Cloud VPC, Cloud Load Balancing |
| **Serverless** | AWS Lambda | Azure Functions, App Service | Cloud Functions, Cloud Run |

---

## Naming Convention Parsing

The analysis engine extracts `product`, `platform`, `environment`, `resource_type`, `region`, and `instance` directly from resource identifiers.

### Default Microsoft CAF Pattern
```
<resource_type>-<workload/product>-<environment>-<region>-<instance>
```

### Custom Naming Conventions
Provide a custom JSON configuration via `--naming` to override token patterns per cloud provider:

```json
{
  "version": 1,
  "default": {
    "delimiter": "-",
    "pattern": ["resource_type", "platform", "product", "environment", "region", "instance"]
  },
  "providers": {
    "aws": {
      "pattern": ["resource_type", "platform", "product", "environment", "instance"]
    }
  }
}
```

### Tag Precedence Modes
- **Fallback (`--naming-mode fallback`, default):** Uses tag values if present; falls back to parsed resource names when tags are missing or `UnTagged`.
- **Override (`--naming-mode override`):** Strictly enforces naming convention values over tag metadata for auditing and standardization.

---

## Quick Start & CLI Usage

### 1. Run Analysis on Single or Multiple Cloud Billing Files

```bash
# Single AWS CUR export
python cur_to_json.py --input mock_aws_cur_iaas_heavy_named20251122.csv --output report_data.json --naming naming_convention.sample.json

# Multi-Cloud ingestion (AWS + Azure + GCP)
python cur_to_json.py \
  --input mock_aws_cur_iaas_heavy_named20251122.csv \
  --input mock_cost_management_billing_50000_with_meter_sku.csv \
  --input mock_gcp_billing.csv \
  --output report_data.json \
  --naming naming_convention.sample.json

# Multi-Month timeline comparison
python cur_to_json.py \
  --input jan.csv --label Jan \
  --input feb.csv --label Feb \
  --input mar.csv --label Mar \
  --output report_data.json
```

### 2. PowerShell Orchestration (Windows / Mac / Linux)

```powershell
# Run analysis, start local HTTP server, and open dashboard in browser
.\run_cur_report.ps1 -CsvPath mock_aws_cur_iaas_heavy_named20251122.csv -NamingPath .\naming_convention.sample.json
```

### 3. Generate Offline Demo Dashboard

```bash
# Injects report_data.json into dashboard.html to produce self-contained dashboard_demo.html
python generate_dashboard.py
```

### 4. Vend Modular Terraform Skeletons

```bash
# Generates modular Terraform configurations per platform/product/environment
python terraform_generator.py --input report_data.json --output-dir terraform_generated
```

---

## File Reference

### Core Engine & Scripts
| File | Description |
| :--- | :--- |
| [`cur_to_json.py`](cur_to_json.py) | **Primary Analysis Engine.** Multi-cloud schema ingestion, service normalization, CAF naming parsing, hardware specs extraction, and JSON reporting. |
| [`billing_provenance.py`](billing_provenance.py) | Source validation, currency and cost-basis checks, observed dates, input fingerprints, and reconciliation metadata. |
| [`terraform_generator.py`](terraform_generator.py) | **IaC Vending Engine.** Generates parameterized, modular Terraform stacks (`main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars`) from discovered infrastructure. |
| [`generate_dashboard.py`](generate_dashboard.py) | **Demo Build Tool.** Compiles `dashboard.html` + `report_data.json` into the standalone `dashboard_demo.html`. |
| [`cur_analysis.py`](cur_analysis.py) | Standalone CLI analysis script for one-off exploration. |
| [`run_cur_report.ps1`](run_cur_report.ps1) | PowerShell orchestration script for automated end-to-end processing and local preview. |

### Visualization & Dashboards
| File | Description |
| :--- | :--- |
| [`index.html`](index.html) | Landing portal directing users to the Demo Dashboard or Live Dashboard. |
| [`dashboard.html`](dashboard.html) | Live interactive dashboard fetching `report_data.json` dynamically via `fetch()`. |
| [`dashboard_demo.html`](dashboard_demo.html) | Standalone offline dashboard with embedded JSON data for zero-setup demos and static hosting. |
| [`dashboard_clean_template.html`](dashboard_clean_template.html) | Historical layout reference; not used by the current build. Edit `dashboard.html` and regenerate the demo. |

### Configuration & Data
| File | Description |
| :--- | :--- |
| [`naming_convention.sample.json`](naming_convention.sample.json) | Configuration schema defining token patterns and delimiter rules per cloud provider. |
| [`report_data.json`](report_data.json) | Complete aggregated JSON output consumed by the dashboard and Terraform generator. |
| [`ROADMAP.md`](ROADMAP.md) | 10-phase strategic FinOps and Platform Engineering roadmap. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history and version milestones. |

---

## Roadmap Overview

See [`ROADMAP.md`](ROADMAP.md) for the full 10-phase strategic plan aligned with the FinOps Foundation lifecycle:

- **Phase 1 (Completed):** Multi-Cloud Foundation & Service Normalization
- **Phase 2:** FOCUS 1.0/1.1 Standard & Scalable Columnar Engine (DuckDB / Parquet)
- **Phase 3:** Temporal Intelligence, MoM Variance & Predictive Forecasting
- **Phase 4:** Tagging Health Scoring (A–F) & Naming Compliance Audits
- **Phase 5:** Rate & Commitment Optimization (RIs, Savings Plans, CUDs, Amortization)
- **Phase 6:** Modern Workloads: Kubernetes (OpenCost) & GenAI/GPU Cost Intelligence
- **Phase 7:** Intelligent Modular IaC Vending & GitOps Drift Detection
- **Phase 8:** Shift-Left CI/CD Cost-as-Code & PR Guardrails
- **Phase 9:** Unit Economics & Sustainability (GreenOps Carbon Tracking)
- **Phase 10:** "Click-to-Fix" Active FinOps & Automated Remediation
