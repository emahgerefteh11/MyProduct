# Strategic Roadmap: Cloud Infrastructure & FinOps Platform

A holistic, multi-phase roadmap aligned with the **FinOps Foundation Framework (Inform, Optimize, Operate)** and tailored for **Leadership, Finance, Platform/Engineering, Governance, and Product** personas.

## Current delivery sequence

1. **Trustworthy reporting (implemented in this iteration):** fix provider/tag attribution; expose report provenance, currency assumptions, source reconciliation, and unallocated spend; reject invalid or incompatible inputs; keep simulated insights in demo mode.
2. **Guided import (implemented):** browser CSV upload, schema/column mapping, full-row validation preview, local report storage, JSON download, and an explicit path from sample data to a customer's first report. Browser reports cover costs and explicit allocation; advanced naming and hardware analysis remain in the Python workflow.
3. **Modern dashboard UI:** substantially refresh the visual design, navigation, charts, and responsive layouts to give the product a polished, modern feel.
4. **Native billing reports and direct cloud connections:** expand AWS CUR/GCP billing-report coverage and add connected-account ingestion that generates and refreshes the reporting dashboard without manual file uploads. Prioritize AWS and GCP, then Azure.
5. **Explain and assign spend:** actual billing-period comparisons, cost-change drivers, ownership rules, and shared-cost allocation.
6. **Adopt IaC, verify, and act:** help organizations move from no IaC to managed infrastructure through billing discovery, cloud configuration enrichment, generated definitions, and reviewed imports; then deliver inventory/utilization-backed findings and changes through their IaC repositories and deployment pipelines.

The longer-term phases below remain directional. Partial FOCUS input handling and labeled multi-file summaries already exist; they do not yet provide full FOCUS conformance or real monthly comparison. The infrastructure builder produces starter templates with assumptions that need review.

## Billing reports and direct cloud connections

**Product requirement:** provide two paths to the same reporting dashboard: upload AWS CUR or GCP billing reports, or connect a hyperscaler account and retrieve its billing data. Both paths should generate results without requiring the user to prepare report JSON or run Python manually.

### Build a dashboard directly from billing reports

- [x] Generate a cost and allocation dashboard from supported AWS CUR and GCP billing CSVs through the guided upload, mapping, and validation flow.
- [ ] Expand native AWS CUR / CUR 2.0 coverage to compressed CSV and Parquet exports, including manifests and partitioned report files.
- [ ] Expand GCP billing-report coverage for native export schemas, nested labels, credits, refunds, and adjustments, with explicit gross/net cost treatment.
- [ ] Handle overlapping exports and revised billing periods without double-counting. Validate that supported file and connected-source inputs produce equivalent cost totals and allocation results.

### Connect a hyperscaler account and generate results

- [ ] **AWS:** guided connection using a cross-account IAM role with an external ID and access scoped to the selected CUR/Data Exports location in S3; support querying the billing export through Athena where appropriate.
- [ ] **GCP:** guided connection to the selected Cloud Billing export dataset/table in BigQuery using federated identity or a scoped service identity, with billing-data read and query permissions.
- [ ] **Azure (follow-on):** connect a billing scope through Cost Management APIs or billing exports using a scoped identity.
- [ ] **Connection onboarding:** select accounts/projects and a billing period, test access, check export prerequisites, preview retrieved data, and generate the dashboard. Explain missing exports, unavailable history, and billing-data delays.
- [ ] **Backend foundation:** authenticated ingestion jobs and private report storage with tenant isolation and narrowly scoped cloud access. Keep cloud credentials out of browser code, browser storage, and the repository. GitHub Pages remains the frontend; connected ingestion requires a backend service.
- [ ] **Refresh and connection management:** manual and configurable scheduled refresh, sync progress, last successful analysis time, actionable errors, and disconnect/revoke controls. Retain the last valid report on a failed sync and visibly mark stale data.
- [ ] **Shared reporting contract:** apply the same currency/cost-basis validation, attribution rules, source provenance, and reconciliation to uploads and connected data; test account boundaries and provider-specific credits/adjustments.

## Modern dashboard UI — To Do

- [ ] Establish a cohesive visual system: modern typography, refined colors, consistent spacing, icons, and reusable components.
- [ ] Redesign navigation and page layouts with clearer hierarchy, more breathing room, and easier access to key cost insights.
- [ ] Present clear **Upload billing reports** and **Connect cloud account** entry points, with connection status and data freshness visible once account ingestion is implemented.
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

## Phase 7: Build on Existing Infrastructure as Code
*Persona Focus: Platform Engineers & DevOps*

**For organizations already using IaC:** connect the organization's repository and Terraform/OpenTofu workspaces, map managed resources to cloud inventory and billing, and propose changes to the code they already maintain. Their existing deployment pipeline remains responsible for applying approved changes. Organizations without IaC use the adoption workflow below. Starter-template generation remains available for new infrastructure.

- [ ] **Read-Only Repository and Workspace Discovery:** identify IaC roots, modules, provider aliases, lockfiles, environments, and state/workspace boundaries. Record the repository commit, workspace, account/project, and collection time for every snapshot.
- [ ] **Customer-Run CI Collector:** start with a collector inside the organization's existing Terraform/OpenTofu pipeline. Read structured state and plan data there and export only allowlisted resource addresses, provider resource IDs, module relationships, and required non-secret attributes. Raw state and plan JSON can contain secrets even when fields are marked sensitive; keep them out of the browser, public artifacts, and this repository.
- [ ] **Resource-to-Code Mapping:** join billing and cloud inventory to provider resource IDs scoped by cloud, account/project, and region; link them to Terraform addresses, module instances, and owning repositories/workspaces. Support `count`/`for_each`, aliases, and multiple states without merging unrelated resources. Show unresolved matches and confidence; names alone are insufficient.
- [ ] **Cost and Ownership Views:** report spend by module, environment, workspace, team, and managed resource. Keep shared charges, credits, and costs without resource IDs visible instead of forcing one-to-one allocation.
- [ ] **Verified Drift and Coverage:** use read-only cloud inventory and refreshed Terraform plan evidence to identify drift or resources outside the connected states. Distinguish desired configuration, the last recorded state, current observations, and historical billing; billing alone does not establish drift or unmanaged status.
- [ ] **Changes to Existing Modules:** use utilization and pricing evidence to propose focused edits to existing module inputs or resource definitions. Preserve module structure, provider constraints, naming, and state addresses. Where code ownership or resource mapping is ambiguous, provide a recommendation for review rather than generating a competing resource definition.
- [ ] **Pull Requests Through Existing CI:** include the evidence, estimated cost impact, and affected resources with each proposed change. Run validation and a fresh plan in the customer's trusted pipeline; highlight replacements, deletions, and data-loss risks. Apply through the organization's existing approval and deployment process, then measure realized savings separately from estimates.
- [ ] **Adopt Unmanaged Resources Deliberately:** implement the billing-report-to-IaC workflow below for resources confirmed outside existing management boundaries. Discovery must not automatically change state or create replacement resources.
- [ ] **New Infrastructure Starter Templates:** retain modular Terraform starter generation for new deployments, using verified SKU and inventory evidence where available and explicit assumptions elsewhere. Filter unused resources only when supporting utilization evidence exists.

**First deliverable:** a read-only Terraform/OpenTofu collector and a cost-by-module/workspace dashboard with source links and unmatched-resource reporting. Add recommendation-to-PR workflows after resource identity and allocation are reliable.

### From billing reports to managed IaC

**Product requirement:** enable organizations with no IaC to use AWS CUR or GCP billing reports as the starting point for bringing their existing infrastructure under Terraform/OpenTofu management.

Billing reports identify services, costs, and some resource IDs; they do not contain a complete deployable configuration. With billing files alone, produce a clearly labeled draft and a list of missing information. Preparing existing resources for adoption requires read-only cloud discovery or equivalent verified inventory supplied by the organization.

- [ ] **Discover and Prioritize:** extract candidate resources from uploaded or connected billing reports and group them by account/project, region, service, application, and spend. Identify stale/deleted resources, charges without resource IDs, and resources already managed elsewhere; do not treat a billing line as proof that a resource still exists or is unmanaged.
- [ ] **Enrich the Configuration:** retrieve current resource settings and dependencies through scoped cloud APIs, including networking, storage attachments, security settings, and identity references where supported. Record source evidence and unresolved attributes. Keep secret values out of generated code and use existing secret references or explicit placeholders.
- [ ] **Generate an Adoption Repository:** create supported Terraform/OpenTofu resource definitions, variables, outputs, provider constraints, dependency references, and import blocks using the actual provider import IDs. Preserve existing names and configuration; show unsupported resource types and missing dependencies for manual resolution. Offer modules after a faithful baseline is established.
- [ ] **Bootstrap the Management Workflow:** guide repository setup, environment boundaries, a protected remote state backend with locking and versioning, scoped CI identity, and plan/review steps for organizations that do not yet have an IaC pipeline.
- [ ] **Preview and Import in Stages:** show the resource-to-code mapping and planned imports for a selected application or environment. Require a reviewed adoption plan with zero resource creates, updates, replacements, or deletes before executing imports through the customer's approved workflow. Verify a subsequent clean plan and retain a state backup. Resolve configuration differences before adoption; optimize after the baseline is stable.
- [ ] **Track Adoption Coverage:** report discovered, eligible, generated, imported, verified, and unresolved resources with ownership and attributable spend. Keep unmatched billing charges visible. Prevent the same resource from being adopted into multiple states.

**First adoption deliverable:** a narrow, tested set of AWS/GCP resource types with a discovery preview, explicit configuration gaps, generated definitions/import blocks, and a guided path to a verified baseline. Full automatic reconstruction from billing data alone is not an acceptance criterion.

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
