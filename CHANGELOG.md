# Changelog

All notable changes to the **MyProduct** infrastructure analysis tool will be documented in this file.

## [Unreleased]

### Added
- Modern workspace UI across the home, dashboard, import and adoption pages: shared design tokens, responsive navigation, redesigned metric cards, tables and report controls, and clear upload/adoption/connection entry points.
- Local cost-mix, platform and compute visualizations with no chart/font CDN requirement; signed cost bars for credits and platform-chart filtering.
- Product search/sorting, scoped product detail dialogs, view deep links/history, mobile navigation with focus handling, skip links and reduced-motion support.
- Chromium checks across 320–1440px widths for demo and imported reports, mobile navigation, safe labels, product details, connection availability, loading and failure states.
- Billing-to-IaC adoption screen and downloadable Python tool: scoped billing candidates, resource selection, gap reporting, and read-only discovery for AWS EC2/EBS and GCP compute instances/zonal disks.
- Terraform/OpenTofu adoption repositories with explicit cloud scope and import IDs, live provider configuration generation, an import-only plan guard, state backup, guarded saved-plan adoption, and clean-plan verification. Allowlisted discovery/import inventory can be reopened locally in the browser.
- Scoped resource evidence in browser/Python reports, including account/project, region/zone, source fingerprints, and charges without resource IDs; flat GCP detailed-export resource paths are retained.
- Adoption engine, cloud command, plan-guard, package download, browser/mobile, and real Terraform provider-schema validation checks.
- Guided browser billing import: multiple CSV files, format detection, editable column mapping, full-row validation, cost/quality preview, and warning acknowledgement before acceptance.
- Private browser report storage, report JSON download, clear-report controls, and dashboard handoff without a backend.
- Import-engine regression tests and Chromium acceptance checks for upload, mapping corrections, validation failures, persistence, downloads, safe labels, and mobile layout.
- Report provenance: actual analysis time, source fingerprints, row counts, currency assumptions, observed dates, cost basis, reconciliation, and unallocated spend.
- Explicit `--currency` handling for missing source currencies; rejection of incompatible currencies and billed/effective cost bases.
- Validation for invalid amounts and duplicate input files, plus Python/JavaScript regression checks and GitHub Actions.

### Fixed
- Native AWS resource-tag columns, Azure JSON tag attribution, FOCUS provider values, and GCP/FOCUS label arrays.
- Report mode no longer generates simulated savings, resource IDs, month-over-month changes, reservation coverage, or arbitrary architecture patterns. Demo simulations are clearly labeled.
- Legacy reports no longer claim today's date as their analysis time or silently assume USD.
- Bundled sample data is explicitly labeled, and new per-product provider attribution is used by both IaC builders; ambiguous provider choices require user selection.
- Dashboard charts and fonts no longer require CDNs, while report tables remain usable independently of the visualization script. Demo generation escapes inline-script terminators and fails when its injection anchor is missing.

### Scope
- Browser imports summarize costs and explicit allocation and feed the separate local IaC adoption workflow. Naming inference, hardware sizing, and new-infrastructure starter generation remain in the Python workflow. Adoption initially covers four resource types and uses customer cloud CLI credentials; hosted connectors, full dependency discovery and repository/CI integration remain future work. Real period comparisons, owner workflows and verified savings remain future work.

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
