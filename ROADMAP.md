# Roadmap: MyProduct Infrastructure Platform

Strategic roadmap aligned with Leadership, Finance, Engineering, and Governance personas.

## Phase 1: The Multi-Cloud Foundation (✅ Completed)
*Persona Focus: Leadership (High-level visibility)*
- [x] **Multi-Cloud Ingestion:** Unified support for AWS, Azure, and GCP billing data.
- [x] **Service Normalization:** Map disparate CSP services (e.g., EC2, VMs, Compute Engine) to unified categories.
- [x] **Resource Counts:** Aggregated counts of VMs and Storage buckets across all clouds.
- [x] **Offline Dashboard:** Portable HTML/JS dashboard for "Anywhere" viewing.
- [x] **Basic Vending:** Generate Terraform skeletons from existing billing footprints.

## Phase 2: Temporal Intelligence & Trends 🚧
*Persona Focus: Leadership & Finance (Growth & Variance)*
- [ ] **Multi-Month Ingestion:** Architecture to load sequential billing files (Jan, Feb, Mar) to establish a timeline.
- [ ] **Month-over-Month (MoM) Growth:**
    - Visual indicators for cost variance ($ and %).
    - Service-level growth (e.g., "Database spend grew 20%").
- [ ] **Trend Visualization:**
    - Line charts for global spend over time.
    - Stacked bar charts for Cloud Service Provider (CSP) mix over time.
- [ ] **Forecasting:** Linear regression to predict end-of-month spend based on current run rate.

## Phase 3: Governance & Compliance
*Persona Focus: Governance (Standards & Tagging)*
- [ ] **Tagging Health Score:**
    - Grade (A-F) per Product based on coverage of mandatory tags (Owner, CostCenter, Env).
    - "Unallocated" cost buckets for untagged resources.
- [ ] **Naming Convention Audit:** Regex-based validation to flag resources violating standard naming patterns.
- [ ] **Compliance Reports:** CSV export of "Offenders" for remediation.

## Phase 4: Engineering Enablement
*Persona Focus: Product Owners & Engineering (Action & Build)*
- [ ] **Utilization-Based Vending:** Only generate Terraform for active/heavy resources (ignore zombies).
- [ ] **Modular Terraform:** Update generator to output industry-standard module usage (e.g., `terraform-aws-modules`) rather than raw resources.
- [ ] **Anomaly Detection:** Statistical detection of SKU distribution outliers (e.g., "Why is Dev using x2large?").
- [ ] **Drift Detection:** "What is billing" vs "What is in Git."

## Phase 5: Unit Economics (The Holy Grail)
*Persona Focus: Finance & Product Owners (Business Value)*
- [ ] **Business Metric Ingestion:** Import "Daily Orders" or "Monthly Active Users" (MAU).
- [ ] **Cost Per Unit:** Correlate cloud spend with business value (e.g., "Cost per Transaction").
- [ ] **Profitability Analysis:** Margin analysis per product feature.

## Phase 6: Sustainability & GreenOps
*Persona Focus: Leadership (ESG) & Governance*
- [ ] **Carbon Estimator:** Calculate CO2e emissions based on region carbon intensity.
- [ ] **Green Optimization:** Recommendations to move workloads to lower-carbon regions.

## Phase 7: "Click-to-Fix" (Active FinOps)
*Persona Focus: Engineering (Remediation)*
- [ ] **Zombie Hunter:** Identify orphaned resources (0 metrics) and queue for deletion.
- [ ] **Spot Readiness:** Analyze workload patterns to recommend Spot Instances.
- [ ] **Ticket Integration:** One-click JIRA/GitHub Issue creation for cost anomalies.
