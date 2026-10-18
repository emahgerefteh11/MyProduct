# Changelog

All notable changes to the **MyProduct** infrastructure analysis tool will be documented in this file.

## [Unreleased]

### Added
- Report provenance: actual analysis time, source fingerprints, row counts, currency assumptions, observed dates, cost basis, reconciliation, and unallocated spend.
- Explicit `--currency` handling for missing source currencies; rejection of incompatible currencies and billed/effective cost bases.
- Validation for invalid amounts and duplicate input files, plus Python/JavaScript regression checks and GitHub Actions.

### Fixed
- Native AWS resource-tag columns, Azure JSON tag attribution, FOCUS provider values, and GCP/FOCUS label arrays.
- Report mode no longer generates simulated savings, resource IDs, month-over-month changes, reservation coverage, or arbitrary architecture patterns. Demo simulations are clearly labeled.
- Legacy reports no longer claim today's date as their analysis time or silently assume USD.
- Bundled sample data is explicitly labeled, and new per-product provider attribution is used by both IaC builders; ambiguous provider choices require user selection.
- Dashboard tables remain available when the chart CDN cannot load. Demo generation escapes inline-script terminators and fails when its injection anchor is missing.

### Scope
- This iteration establishes trustworthy reporting. Guided import, real period comparisons, owner workflows, verified savings, and validated infrastructure changes remain future work.

## [v0.2.0] - 2026-02-17

### Added
- **Multi-Cloud Support**: Added schema detection and normalization for **GCP** (Google Cloud Platform) and **Azure** billing exports.
- **Terraform Vending**: Created `terraform_generator.py` to automatically generate `main.tf` skeletons per platform/product/environment based on discovered billing usage.
- **Offline Dashboard**: Added `dashboard.html` (formerly `report.html`) to the root directory for GitHub Pages deployment. It now loads `report_data.json` directly without requiring a local Python server.
- **GCP Mock Data**: Added `mock_gcp_billing.csv` ($100k/month scale) to validate multi-cloud aggregation.
- **Azure Integration**: Integrated `mock_cost_management_billing_50000_with_meter_sku.csv` for full AWS/GCP/Azure reporting.

### Changed
- **Service Normalization**: Refined `SERVICE_CATEGORY_RULES` in `cur_to_json.py` to prioritize Storage and Database detection over generic Compute matching (fixes Azure Managed Disk misclassification).
- **Dashboard UI**: Updated the Storage Share card to explicitly state "Block + Object" storage, reflecting the new multi-cloud normalization.
- **Project Structure**: Promoted the rich UI `MyProduct/report.html` to `dashboard.html` in the root.

### Fixed
- **GCP Label Parsing**: Fixed issue where GCP labels (Product/Environment) were read as raw JSON strings. Added parser to extracting values from the `project:labels:{...}` format.
- **GCP VM Sizing**: Added heuristic logic to estimate GCP vCPU/RAM based on "Instance Core" usage types.
- **Dashboard Paths**: Fixed dashboard resource loading to work from the repository root.

## [v0.1.0] - Initial Release
- Basic AWS CUR parsing.
- JSON report generation.
- CLI analysis script.
