# Roadmap: MyProduct Infrastructure Platform

Strategic roadmap based on key personas: Leadership, Finance, Engineering, and Governance.

## Phase 1: The Multi-Cloud Foundation (✅ Completed)
*Goal: Unified visibility across AWS, Azure, and GCP.*
- [x] **Multi-Cloud Ingestion:** Support for AWS CUR, Azure Cost Export, and GCP Billing.
- [x] **Service Normalization:** Map disparate CSP services to unified categories (Compute, Storage, etc.).
- [x] **Offline Dashboard:** Portable HTML/JS dashboard for "Anywhere" viewing.
- [x] **Basic Vending:** Generate Terraform skeletons from existing billing footprints.

## Phase 2: Temporal Intelligence (Leadership & Finance) 🚧
*Goal: Tracking trends, growth, and direction over time.*
- [ ] **Multi-Month Support:** Ingest multiple billing periods simultaneously.
- [ ] **MoM Metrics:** Calculate Month-over-Month variance ($ and %) for top-level KPIs.
- [ ] **Trend Visualization:** Line charts for Cost and Resource Counts over time.
- [ ] **Forecast:** Simple linear projection of end-of-month spend.

## Phase 3: Governance & Compliance (Governance)
*Goal: Enforcing standards and visibility.*
- [ ] **Tagging Health Score:** Quantitative metric (% tagged) per Product/Platform.
- [ ] **Compliance Reports:** CSV export of resources violating tagging policies.
- [ ] **Naming Validation:** Regex-based audit of resource names against defined conventions.
- [ ] **Budget Alerts:** Visual flags for budget overruns.

## Phase 4: Engineering Enablement (Product Owners)
*Goal: Closing the loop from "Running" to "IaC".*
- [ ] **Modular Terraform:** Update generator to use standardized modules (e.g., `terraform-aws-modules`).
- [ ] **Right-Sizing Insights:** Highlight underutilized SKUs (e.g., huge VMs with low cost efficiency).
- [ ] **Anomaly Detection:** Statistical detection of spend spikes at the Service/Product level.
- [ ] **Workload Scoping:** Better grouping of resources into logical "Stacks" for Terraform export.
