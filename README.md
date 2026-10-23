# Cloud Cost Atlas (MyProduct)

[![Multi-Cloud](https://img.shields.io/badge/Multi--Cloud-AWS%20%7C%20Azure%20%7C%20GCP-blue)](https://github.com/emahgerefteh11/MyProduct)
[![FinOps](https://img.shields.io/badge/FinOps-Foundation%20Aligned-green)](https://www.finops.org/)
[![IaC](https://img.shields.io/badge/IaC-Terraform%20Vending-purple)](https://www.terraform.io/)
[![Static Dashboard](https://img.shields.io/badge/Dashboard-HTML5%20%2F%20JS-orange)](dashboard.html)

**Cloud Cost Atlas** is a multi-cloud infrastructure intelligence and FinOps tool. It ingests supported billing CSV formats from **AWS**, **Microsoft Azure**, and **Google Cloud Platform (GCP)**, normalizes services into a unified taxonomy, extracts hardware sizing and naming conventions, generates interactive dashboards, and creates Terraform starter templates based on reported cloud footprints. Generated infrastructure requires review and validation before deployment.

## Guided billing import

Open [Import billing](import.html) from the home page or dashboard. This works on GitHub Pages without a backend or cloud credentials.

Supported **AWS CUR and GCP billing CSVs can directly generate the reporting dashboard** through this flow; no manual JSON conversion or Python command is required. Broader native export support, including compressed/Parquet CUR files and GCP credit handling, is tracked in the [billing-source roadmap](ROADMAP.md#billing-reports-and-direct-cloud-connections).

1. Choose uncompressed UTF-8 CSV files. The browser importer accepts up to **10 files, 20 MiB combined, and 100,000 rows total**. Comma, semicolon, and tab separators are supported.
2. Review the detected format and column mappings for each file. AWS CUR, Azure Cost Management, GCP billing exports, and FOCUS-style headers are recognized; custom exports can be mapped manually. Confirm the cost column, provider fallback, and billed/effective cost basis. A currency assumption is optional and only fills missing currencies.
3. Validate every row and review the preview: totals, source row counts, observed dates, top services, unallocated spend, reconciliation, and warnings. Invalid amounts, malformed row widths, mixed currencies or cost bases, reversed date ranges, and identical whole-file duplicates block acceptance. Correct the CSV or mappings and validate again.
4. Acknowledge the warnings, then **Open dashboard** or **Download report JSON**. The dashboard opens in report mode, with no simulated savings or trends.

Files are parsed and validated in a Web Worker on the device. Only an accepted report is saved in this browser's IndexedDB when **Open dashboard** is clicked; source CSV files are not persisted or sent to a server. The accepted report takes precedence over the site's bundled report and survives a refresh. The dashboard's **Download report** and **Clear imported report** controls let you keep a copy or remove it. Demo mode remains independent. Browser storage is specific to this site, browser, and device; private browsing, clearing site data, or changing devices can remove access to the saved report. Download JSON to keep a portable copy.

The browser report covers **costs, explicit product/platform/environment columns or JSON tags, providers, and resource identifiers as supplied**. It does not infer names, hardware sizing, usage capacity, or utilization, and infrastructure generation is unavailable for browser reports. Use the Python analyzer for naming rules, supported hardware/SKU inference, larger files, or infrastructure starter generation. Separate GCP credit fields are not subtracted automatically; map an appropriate net-cost column if required. FOCUS header recognition is not full FOCUS conformance. Duplicate checking detects identical file contents, not overlapping rows in distinct exports. Monetary sums use exact decimal arithmetic before rounding totals for display; no currency conversion is performed.

Use a current browser over **HTTPS or localhost**. Opening `import.html` directly through `file://` is not supported. For local use, run `python -m http.server 8000` in the repository and visit `http://localhost:8000/import.html`. The import page has no third-party dependencies; the dashboard still loads charts and fonts from CDNs, with tables available if those resources cannot load.

## Direct cloud account connections (planned)

Direct hyperscaler account connections are **planned, not yet implemented**. The target flow is to connect AWS or GCP, select the billing source and period, and generate or refresh the same dashboard automatically; Azure follows. This requires an authenticated backend for cloud access and ingestion. AWS CUR exports in S3 and GCP Cloud Billing exports in BigQuery are the initial connected sources. See the [connection requirements](ROADMAP.md#connect-a-hyperscaler-account-and-generate-results).

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

Real period comparison, verified savings, and infrastructure-change workflows remain upcoming iterations. The current multi-file summaries do not establish comparable monthly periods.

New reports preserve provider attribution per platform/product. The IaC builders use that evidence instead of inferring a cloud from the team name. The browser builder blocks ambiguous providers; the CLI requires an explicit `--provider` choice for those products. Legacy reports retain heuristic provider selection. Templates remain unvalidated starter configurations.

For organizations with existing Terraform/OpenTofu, the planned workflow is to connect their repositories and workspaces, reconcile managed resources with cloud inventory and billing, and propose changes to their existing modules through pull requests and their normal CI/CD process. Start with read-only cost attribution by module/workspace; state/plan collection should run inside the customer's environment and export only allowlisted metadata. Existing-infrastructure discovery, reconciliation, and PR generation are **not yet implemented**. See [Build on Existing Infrastructure as Code](ROADMAP.md#phase-7-build-on-existing-infrastructure-as-code).

### Regression checks

The analysis engine and its tests use the Python standard library. Dashboard behavior checks use Node.js and a minimal DOM harness; they do not require a browser or CDN access.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/dashboard.test.cjs
node tests/import.test.cjs
python generate_dashboard.py
```

GitHub Actions runs these checks, verifies that the generated demo matches `dashboard.html` and `report_data.json`, and runs Chromium acceptance tests for the guided import. To run the browser checks locally:

```bash
python -m pip install playwright==1.56.0
python -m playwright install chromium
python tests/browser_import.py
```

The browser tests start a local server and block third-party requests. Screenshots are written to `test-artifacts/` (ignored by Git); CI retains them for seven days.

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
| [`index.html`](index.html) | Landing portal for guided billing import, sample demo, and report dashboard. |
| [`import.html`](import.html), [`import.js`](import.js), [`import.css`](import.css) | Guided file selection, editable mappings, validation preview, and report acceptance. |
| [`billing_import.js`](billing_import.js), [`billing_worker.js`](billing_worker.js) | Browser CSV parsing, schema detection, validation, exact decimal sums, and cost aggregation in a worker. |
| [`report_store.js`](report_store.js) | Browser-only IndexedDB storage for one accepted report. |
| [`dashboard.html`](dashboard.html) | Interactive dashboard loading an accepted browser import first, then the site's `report_data.json` when no import is saved. |
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
