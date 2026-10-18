# Strategic Roadmap: Cloud Infrastructure & FinOps Platform

A holistic, multi-phase roadmap aligned with the **FinOps Foundation Framework (Inform, Optimize, Operate)** and tailored for **Leadership, Finance, Platform/Engineering, Governance, and Product** personas.

## Current delivery sequence

1. **Trustworthy reporting (implemented in this iteration):** fix provider/tag attribution; expose report provenance, currency assumptions, source reconciliation, and unallocated spend; reject invalid or incompatible inputs; keep simulated insights in demo mode.
2. **Guided import:** billing-file upload, schema/column mapping, validation preview, and an explicit path from sample data to a customer's first report.
3. **Modern dashboard UI:** substantially refresh the visual design, navigation, charts, and responsive layouts to give the product a polished, modern feel.
4. **Explain and assign spend:** actual billing-period comparisons, cost-change drivers, ownership rules, and shared-cost allocation.
5. **Verify and act:** inventory/utilization-backed findings, savings status and measurement, and reviewed changes to existing infrastructure.

The longer-term phases below remain directional. Partial FOCUS input handling and labeled multi-file summaries already exist; they do not yet provide full FOCUS conformance or real monthly comparison. The infrastructure builder produces starter templates with assumptions that need review.

## Modern dashboard UI — To Do

- [ ] Establish a cohesive visual system: modern typography, refined colors, consistent spacing, icons, and reusable components.
- [ ] Redesign navigation and page layouts with clearer hierarchy, more breathing room, and easier access to key cost insights.
- [ ] Modernize summary cards, charts, tables, filters, and drill-down interactions while keeping report provenance and demo labels visible.
- [ ] Make the dashboard work comfortably on desktop, tablet, and mobile screens.
- [ ] Polish loading, empty, error, hover, and focus states; support keyboard navigation, readable contrast, and reduced motion.
- [ ] Validate the redesign in a real browser across screen sizes using both sample data and imported reports.

---

## Phase 1: Multi-Cloud Foundation & Ingestion Parity (✅ Completed)
*Persona Focus: Leadership & Platform Engineering (Multi-Cloud Visibility)*
- [x] **Unified Multi-Cloud Ingestion:** Ingestion support for AWS Cost and Usage Reports (CUR), Azure Cost Management exports, and GCP Standard Billing exports.
- [x] **Cross-Cloud Service Normalization:** Mapping disparate CSP service lines into 8 unified functional families (Compute/VM, Block Storage, Object Storage, Database, Cache, Container, Networking, Serverless).
- [x] **Hardware & Sizing Introspection:** Aggregation of vCPU, RAM GiB, and storage GB-Mo from cloud billing SKUs.
- [x] **Zero-Dependency Offline Dashboard:** Portable HTML5/JS single-page application with dual-mode architecture (Live dynamic fetch + embedded demo).
- [x] **Initial IaC Vending:** Automatic generation of Terraform baseline skeletons from discovered billing topology.

---

## Phase 2: Open Standards & High-Performance Data Pipeline 🚀 *(Next Up)*
*Persona Focus: Platform Engineering & Enterprise Architecture*
- [ ] **FOCUS (FinOps Open Cost & Usage Specification) 1.0/1.1 Adoption:**
  - Standardize internal data schemas on the Linux Foundation / FinOps Foundation FOCUS open specification (`ChargeCategory`, `BilledCost`, `EffectiveCost`, `PricingQuantity`, `ProviderName`, `ResourceId`, `ServiceName`, `SubAccountId`).
  - Native export capability to FOCUS-compliant CSV/Parquet for interoperability with standard FinOps tooling.
- [ ] **Scalable Columnar Query Engine (DuckDB & Parquet):**
  - Replace memory-bound CSV parsing with an embedded **DuckDB** / **Apache Arrow** query engine to process 10M+ row enterprise billing files (5GB–50GB+) in seconds.
  - Automatic conversion of raw CSV dumps to partitioned, compressed Parquet files for fast query execution.
- [ ] **Multi-Currency & FX Normalization:**
  - Ingestion of non-USD billing currencies (EUR, GBP, JPY, CAD) with automated or configurable FX exchange rate conversion.

---

## Phase 3: Temporal Intelligence, Trends & Forecasting 🚧
*Persona Focus: Leadership & Finance (Growth, Budgeting & Variance)*
- [ ] **Multi-Month Timeline Ingestion:**
  - Multi-period sequential billing ingestion (e.g. Q1, Q2, H1) to establish longitudinal cost trends.
- [ ] **Month-over-Month (MoM) Variance Engine:**
  - Visual delta indicators for spend variance ($ and %) by platform, product, service family, and account.
  - Drill-down variance explanations (e.g., "Platform A Database spend grew +$14.2k due to Aurora cluster scaling").
- [ ] **Advanced Spend Visualization:**
  - Historical spend trajectory line charts and stacked multi-cloud CSP mix breakdowns over time.
- [ ] **Predictive Forecasting & Run-Rate Modeling:**
  - Exponential smoothing and linear regression algorithms to project end-of-month (EOM) and quarterly cloud run-rates based on consumption velocity.
  - Budget vs. Actual tracking with predictive burn-rate threshold alerts.

---

## Phase 4: Governance, Tagging Quality & Compliance
*Persona Focus: Governance, Security & FinOps Practitioners*
- [ ] **Tagging Health & Coverage Scoring:**
  - Automated A–F grade per Platform/Product based on mandatory tag coverage (`Owner`, `CostCenter`, `Environment`, `Service`).
  - Identification and isolation of "Unallocated / Untagged" spend pools.
- [ ] **Naming Convention Audit & Enforcement:**
  - Regex-based rule engine validating resource names against standard patterns (e.g., Microsoft Azure CAF, AWS Enterprise tagging).
  - Identification of naming convention outliers and misclassified resources.
- [ ] **Compliance & Remediation Exports:**
  - Actionable CSV and JSON reports detailing non-compliant resources for automated ticketing or team notifications.

---

## Phase 5: Rate & Commitment Optimization (FinOps "Optimize")
*Persona Focus: Finance, Procurement & FinOps Practitioners*
- [ ] **Commitment Portfolio Visibility:**
  - Tracking of AWS Savings Plans / Reserved Instances (RIs), Azure Reservations / Savings Plans, and GCP Committed Use Discounts (CUDs).
- [ ] **Coverage & Utilization Analytics:**
  - **Commitment Coverage:** % of eligible compute/database hours running under discount commitments vs On-Demand rates.
  - **Commitment Utilization:** Identification of underutilized or expiring commitments (avoiding wasted reservation dollars).
- [ ] **Effective Savings Rate (ESR) Computation:**
  - True financial discount rate achieved across all clouds compared to list On-Demand rates.
- [ ] **Unblended vs. Amortized Cost Views:**
  - Support for amortized cost accounting (spreading upfront commitment payments evenly across consumption).

---

## Phase 6: Modern Workload FinOps: Kubernetes & GenAI
*Persona Focus: Engineering Leads, ML Platform Engineers & DevOps*
- [ ] **Container & Kubernetes Cost Allocation (K8s FinOps):**
  - Integration with **OpenCost** / **Kubecost** or Prometheus metrics to disaggregate monolithic EKS, AKS, and GKE cluster node costs into Kubernetes namespaces, daemonsets, pods, and microservices.
  - Allocation of shared cluster overhead and idle capacity costs.
- [ ] **GenAI & GPU Workload Intelligence:**
  - Dedicated tracking for high-cost AI infrastructure (NVIDIA H100, A100, L4, T4 GPU instances).
  - Ingestion and cost attribution for managed AI services (AWS Bedrock / SageMaker, Azure OpenAI, GCP Vertex AI).
  - Cost-per-token and cost-per-inference unit metrics.

---

## Phase 7: Intelligent Infrastructure as Code (IaC) Vending & Drift
*Persona Focus: Platform Engineers & DevOps*
- [ ] **SKU-Aware Modular Terraform Vending:**
  - Generate production-ready Terraform modules (`main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars`) mapped to exact discovered hardware SKUs, storage tiers, and network topology.
  - Modular support for verified ecosystem modules (e.g., `terraform-aws-modules`).
- [ ] **Utilization-Filtered Vending:**
  - Filter out zombie/orphaned resources from generated IaC skeletons (vending only active, healthy infrastructure).
- [ ] **GitOps Drift Detection ("Billing vs. Git"):**
  - Compare discovered active cloud resources against committed Terraform/OpenTofu state files in Git to identify shadow IT and unmanaged cloud sprawl.

---

## Phase 8: Shift-Left "Cost-as-Code" & CI/CD Integration
*Persona Focus: Software Developers & Platform Engineering*
- [ ] **Pull Request Cost Estimation (Shift-Left FinOps):**
  - GitHub Actions / GitLab CI runner that analyzes Terraform plan outputs on pull requests and computes the delta against current platform baselines.
- [ ] **Policy-as-Code Guardrails:**
  - Integration with Open Policy Agent (OPA) / Conftest to block PRs that exceed team budget thresholds or provision banned/oversized instance types.
- [ ] **FinOps Slack/Teams Notifications:**
  - Automated deployment cost summaries delivered to squad channels upon merged infrastructure PRs.

---

## Phase 9: Unit Economics & Sustainability (GreenOps)
*Persona Focus: Executive Leadership (CFO/CTO/ESG) & Product Management*
- [ ] **Unit Economics & Business Metric Ingestion:**
  - Ingest business metrics (e.g., Daily Active Users, Checkout Transactions, API Calls, ML Training Runs).
  - Calculate real-time **Cost per Business Unit** (e.g., "$0.0034 per payment transaction").
- [ ] **Product Feature Margin Analysis:**
  - Correlate cloud infrastructure spend with product revenue to determine gross margins by platform feature.
- [ ] **GreenOps & Carbon Estimator (CO2e):**
  - Estimate embodied and operational carbon emissions (MTCO2e) based on regional grid carbon intensity and hardware power profiles.
  - Recommend greener region migrations (e.g., shifting non-latency-sensitive batch compute to low-carbon regions).

---

## Phase 10: "Click-to-Fix" Active FinOps & Automated Remediation
*Persona Focus: Engineering & Operations*
- [ ] **Zombie Hunter & Idle Resource Killer:**
  - Automated detection of unattached EBS/Managed Disks, unassociated Elastic IPs, orphaned snapshots, and 0-metric idle compute.
- [ ] **Spot & Graviton/ARM Migration Advisor:**
  - Workload analysis to identify stateless compute candidates for Spot instances or AWS Graviton / Azure ARM processor migration (saving 20–40%).
- [ ] **ITSM Ticketing Integration:**
  - One-click Jira, ServiceNow, or GitHub Issue generation with pre-populated resource IDs, cost savings estimates, and remediation playbooks.

---

## Persona Value Matrix

| Persona | Core Benefits & Key Features |
| :--- | :--- |
| **Executive Leadership (CEO/CTO/CFO)** | Single-pane-of-glass multi-cloud cost visibility, cloud mix distribution, ESG carbon tracking, unit economics margin analysis. |
| **Finance & Procurement** | MoM variance reporting, predictive run-rate forecasting, commitment coverage (RI/SP/CUD), amortized cost accounting. |
| **Platform & DevOps Engineering** | Automated Terraform module vending, GitOps drift detection, shift-left PR cost estimation, Kubernetes container cost allocation. |
| **Governance & Security** | Tagging health grades (A–F), CAF naming convention audit, orphan resource detection, data masking and team RBAC. |
| **Product Owners** | Clear cost-per-feature and cost-per-user metrics, enabling data-driven roadmap prioritization and pricing strategy. |
