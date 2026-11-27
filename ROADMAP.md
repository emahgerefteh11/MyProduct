# Strategic Roadmap: Cloud Infrastructure & FinOps Platform

A holistic, multi-phase roadmap aligned with the **FinOps Foundation Framework (Inform, Optimize, Operate)** and tailored for **Leadership, Finance, Platform/Engineering, Governance, and Product** personas.

## Current delivery sequence

1. **Trustworthy reporting (implemented in this iteration):** fix provider/tag attribution; expose report provenance, currency assumptions, source reconciliation, and unallocated spend; reject invalid or incompatible inputs; keep simulated insights in demo mode.
2. **Guided import (implemented):** browser CSV upload, schema/column mapping, full-row validation preview, local report storage, JSON download, and an explicit path from sample data to a customer's first report. Browser reports cover costs and explicit allocation; advanced naming and hardware analysis remain in the Python workflow.
3. **Modern dashboard UI (implemented):** shared visual system across the home, dashboard, import and adoption screens; responsive navigation, local charts, product drill-downs, accessible controls and polished report states.
4. **Native billing reports and direct cloud connections:** expand AWS CUR/GCP billing-report coverage and add connected-account ingestion that generates and refreshes the reporting dashboard without manual file uploads. Prioritize AWS and GCP, then Azure.
5. **Explain and assign spend:** actual billing-period comparisons, cost-change drivers, ownership rules, and shared-cost allocation.
6. **Adopt IaC, verify, and act:** help organizations move from no IaC to managed infrastructure through billing discovery, cloud configuration enrichment, generated definitions, and reviewed imports; then deliver inventory/utilization-backed findings and changes through their IaC repositories and deployment pipelines.

The longer-term phases below remain directional. Partial FOCUS input handling and labeled multi-file summaries already exist; they do not yet provide full FOCUS conformance or real monthly comparison. The infrastructure builder produces starter templates with assumptions that need review.

## Billing reports and direct cloud connections

### AWS MVP delivery checkpoint

[Operator preflight and live release checks](docs/aws-pilot.md)

- **Synthetic acceptance:** commit [`edb3ade`](https://github.com/emahgerefteh11/MyProduct/commit/edb3adecb0fd8e82d6746103949e8319563bb092) records the classic CUR fixture, credential-free manifest adapter and regression coverage on the billing-parity branch. The local Node suite reconciles the browser worker and gzip connector at $100 USD net ($70 Payments, $30 unallocated); retries remain $100, the correction gives $105 with $35 unallocated, and a failed refresh preserves the accepted report and timestamp.
- **Additional local parity:** the same test file covers multipart order, a missing manifest-listed part, incompatible currency, out-of-period rows, unsupported parts, and tenant/source/period scoping. The fixture is synthetic. No live AWS access or deployment happened.
- **Scope:** classic, comma-delimited, UTF-8 AWS CUR CSV gzip parts named in a manifest, with per-row matching billing dates, explicit currency, and billed unblended cost. Parquet, effective/mixed-basis input, native CUR 2.0 manifests and alerting are excluded. The adapter stores a single validated in-memory snapshot per tenant, source and period. The kept-versions fixture now reads the current period manifest and validates its timestamped assembly ID, reads only the parts for that assembly, and verifies the manifest again before acceptance. Replaying the old $100 assembly after accepting $105, or advancing the manifest during a refresh, does not overwrite the accepted report or its timestamp. Missing listed parts prevent acceptance; an upstream manifest that omits an expected part, live AWS behavior, overwriting in place, and authenticated server-side tenant isolation still require separate evidence. See [AWS report versions](https://docs.aws.amazon.com/cur/latest/userguide/understanding-report-versions.html) and [current report lookup](https://docs.aws.amazon.com/cur/latest/userguide/view-latest-cur.html).
- **Pilot handling of split and finalized exports:** A manifest-listed gzip CSV with only a header matching a populated part is included in revision integrity checks but excluded from spend and coverage. Missing, corrupt, zero-byte, malformed, incompatible and all-empty parts or assemblies still fail without publishing. For upload parity, select only the populated, decompressed CSV parts and record the omitted empty part separately. Confirm the real export’s empty-part representation; see [AWS empty-partition guidance](https://docs.aws.amazon.com/cur/latest/userguide/troubleshooting.html#report-data-partitions-empty). Atlas date coverage is not invoice finalization: compare a finalized period using the invoice-ID signal without logging IDs. A same-assembly, same-cost-basis comparison for an open month is provisional. [AWS finalization](https://docs.aws.amazon.com/cur/latest/userguide/view-finalized-cur.html).
- **Cost basis evidence:** Compare the exact mapped cost column and report provenance; the generic billed label is not sufficient. The Legacy CUR connector defaults to `lineItem/UnblendedCost`. Use that same column for initial upload parity, even if the export also has `lineItem/NetUnblendedCost`. A synthetic discount proves those two valid uploads can disagree without a connector bug.
- **Exact ownership mapping:** The async connector can read a versioned ownership column from trusted source configuration. It validates the original case across parts, records the selected mapping, and can recompute the same bytes only under a newer active mapping. A configuration change during validation, unsupported column, or stale mapping cannot replace the accepted report. Use the same explicit column for the manual upload; overlapping tag-header case remains unsupported.
- **Verify:** run `node --test tests/cur_parity.test.cjs`; the browser import suite also has a synthetic fixture assertion. Local Python and Node regressions and generated-page checks passed for the first commit. Browser execution and hosted CI are separately tracked; do not count this checkpoint as a live pilot.
- **Next:** confirm the existing pilot export variant and propose the private AWS ingestion architecture. Before deploying a connected application, choose an authenticated host and prove that disconnecting invalidates in-flight publication. The [main browser and regression run](https://github.com/emahgerefteh11/MyProduct/actions/runs/45802565941) is the verified baseline. The live pilot needs approved account access and production approval.


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

### Private AWS ingestion: publication contract

- A local, executable SQLite contract (`tests/test_ingestion_publish_contract.py`) records the generation at job capture and checks it in the same write that publishes the report. It verifies a validated $105 result cannot displace the last accepted $100 report or its timestamp after a disconnect or after reconnecting. A fresh job can publish after reconnect. It also checks that two tenants with the same source and period use separate report records.
- The test contract also gates publication on the active mapping version and the ordered kept-versions assembly. A validated older $100 job or same-assembly retry cannot replace the $105 result or move its timestamp. An updated mapping can reprocess the current assembly only if the validated input fingerprint still matches. A stale mapping cannot publish even a newer assembly. The connector still must verify the active manifest separately.
- A loopback-only HTTP fixture (`tests/test_ingestion_access_contract.py`) demonstrates the private read contract against the publication store. Job and report reads use the tenant resolved from an opaque fixture session; tenant hints in requests are ignored, another tenant’s job looks missing, expired or revoked sessions are denied, and private responses are not cacheable. These tests do not implement production authentication.
- The server-side private read adapter (`server/private_billing_read.cjs`) accepts a verified session resolver and scoped storage. It returns non-cacheable results, exposes only safe job status, and binds reads to the resolved tenant. Integration tests pass actual fixture connector reports into the browser gate and check account switch, session revocation, and upload independence. No public route, provider, storage engine, or deployment is introduced.
- The server stream adapter (`server/bounded_s3_read.cjs`) supports a scoped injected S3 reader and enforces compressed byte bounds, closes rejected bodies, and distinguishes conditional manifest movement from transport errors. The existing connector separately bounds decompression. The async kept-manifest path now stages only validated assembly keys within an aggregate compressed budget, validates expanded data, and verifies the manifest again before commit. Fake-S3 integration checks accept $100 then $105, preserve results on failed reads, and keep a later-finishing old worker from reverting a newer commit. This remains an in-memory and synthetic path; a hosted transaction, verified AWS role/provider and live AWS are still outstanding.
- The scoped S3 bridge (`server/scoped_cur_s3.cjs`) accepts only reads for one preconfigured source and billing period, passes the expected bucket owner and optional manifest ETag to an injected GetObject client, and never enumerates the bucket. Manifest-listed keys remain checked by the connector. The hosting integration must still assume an approved read-only role and supply the real client.
- Before choosing hosted storage or serving the full private report in a response, measure encoded bytes including resource detail and the host envelope. Synthetic high-cardinality reports exceeded 6 MiB at 25,000 unique resources despite modest CSV sizes. If necessary, evaluate a bounded overview and separate authorized detail rather than silently dropping inventory.
- The synthetic durable job result is written in the same transaction as the accepted snapshot. A lost success response does not redate or republish the report on retry, even when a later job has replaced latest; a failed job-result write rolls back the snapshot change. A stable job ID cannot be reused for a different revision, mapping, or scope. Hosted storage remains undecided.
- The vendor-neutral refresh runner (`server/cur_refresh_job.cjs`) resolves a scoped internal job, checks terminal outcome before AWS access, verifies the scheduled period and active connection/mapping during parsing, and hands the candidate to an injected atomic publisher. Fixture integration covers a lost acknowledgement without another S3 fetch and a disconnect before publication. The queue, verified credentials, and durable production publisher are not wired.
- [Restart acceptance](https://github.com/emahgerefteh11/MyProduct/commit/0920dd47ac65f0c5e29bee0454d367ed45c16112) composes the actual refresh runner, scoped fake-S3 reads and private read handler over independent connections to a temporary on-disk database. The fixture preserves its $105 correction and job completion after a lost confirmation and reopen; replay makes no object reads and cannot revert a later $110 report. Another tenant cannot read the report or job. [Main CI](https://github.com/emahgerefteh11/MyProduct/actions/runs/45945190528) passed regression, Chromium and Terraform. This is a graceful local reopen test, not a process crash, host failure, or deployed pilot.
- This is a storage contract under synthetic data, not an authenticated service, a production database choice, or live AWS evidence. The hosted implementation must derive tenant identity from authentication, fence updates transactionally across workers, preserve revision-order behavior, and pass cross-tenant authorization and revoked-access tests.
- The candidate connected view gate (`report_view_session.js`) fences late responses and requires explicit selection of connected or browser-local data; the browser fixture runs against real IndexedDB with simulated connected responses. This gate is not wired to the public dashboard or a deployed private endpoint.
- The [transactional action rehearsal](https://github.com/emahgerefteh11/MyProduct/actions/runs/45989092060) tests the private refresh and disconnect POST handler, scoped worker and read handler together. A retried refresh queues one durable job; permission revocation at write time is denied; disconnect during a newer refresh cancels its publication and preserves the accepted $105 report and timestamp. No live AWS calls or customer-facing write route are enabled.
- The [loopback Chromium rehearsal](https://github.com/emahgerefteh11/MyProduct/actions/runs/45973346160) sends an actual request to the private report handler using a verified fixture session, reads the $105 report, denies other-tenant and revoked sessions, and leaves the $100 manual upload intact in browser storage. This proves the integration with synthetic data only; a real identity provider, private host and approved AWS pilot remain outstanding.
- The [browser action rehearsal](https://github.com/emahgerefteh11/MyProduct/actions/runs/46006547164) uses the same protected operations and test store through a loopback HTTP server. The accepted amount and timestamp stay visible while a job is queued or running; completion triggers a fresh read. A verified viewer can still read, a revoked session loses access, and a stopped revision cannot replace the last good report. These controls are test-only and are not exposed on the public demo.
- Connected entry points should explicitly request a connected report, never silently fall back to a browser-local import or replace the existing browser-local slot. After sign-out or a tenant switch, delayed responses and browser back navigation must not show another tenant's data. The existing manual upload remains independent.
- Next: select an authenticated hosting path, verify the existing pilot export format and ownership mapping, and implement the smallest private vertical slice without granting AWS access or deploying until approved.

## Modern dashboard UI — Implemented

- [x] Establish a cohesive visual system: modern typography, refined colors, consistent spacing, icons, and reusable components.
- [x] Redesign navigation and page layouts with clearer hierarchy, more breathing room, and easier access to key cost insights.
- [x] Present clear **Upload billing reports** and **Connect cloud account** entry points. Connection options explicitly show planned status; actual analysis time and observed usage are visible. Connected sync status will follow backend ingestion.
- [x] Modernize summary cards, charts, tables, filters, and drill-down interactions while keeping report provenance and demo labels visible.
- [x] Make the dashboard work comfortably on desktop, tablet, and mobile screens.
- [x] Polish loading, empty, error, hover, and focus states; support keyboard navigation, readable contrast, and reduced motion.
- [x] Validate the redesign in a real browser across screen sizes using both sample data and imported reports.

The dashboard now uses local charts with signed-cost handling for credits, chart-to-platform filtering, persistent product search/sorting, product detail dialogs, deep links, and a keyboard-accessible mobile navigation drawer. Shared assets require no chart or font CDN. Browser checks cover 320–1440px layouts, report/demo separation, loading and failure states, safe labels, dialogs and focus handling. This UI release does not implement hosted cloud ingestion or verified optimization findings.

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

**Implemented first release:** [Adopt into IaC](adoption.html) now discovers supported candidates from current browser/Python reports, keeps account/project and region scope, shows gaps and unattributed charges, and downloads a working local adoption tool. Read-only AWS/GCP discovery supports EC2 instances, EBS volumes, GCP compute instances and zonal disks. The tool creates provider configuration/import blocks, uses Terraform/OpenTofu to generate live resource definitions, rejects plans with infrastructure changes, imports the reviewed saved plan, backs up existing state, and requires a clean subsequent plan. Discovery/import inventory can be reopened in the browser. This release uses customer CLI credentials; hosted connectors and the broader requirements below remain follow-up work.

**Product requirement:** enable organizations with no IaC to use AWS CUR or GCP billing reports as the starting point for bringing their existing infrastructure under Terraform/OpenTofu management.

Billing reports identify services, costs, and some resource IDs; they do not contain a complete deployable configuration. With billing files alone, produce a clearly labeled draft and a list of missing information. Preparing existing resources for adoption requires read-only cloud discovery or equivalent verified inventory supplied by the organization.

- [ ] **Discover and Prioritize:** extract candidate resources from uploaded or connected billing reports and group them by account/project, region, service, application, and spend. Identify stale/deleted resources, charges without resource IDs, and resources already managed elsewhere; do not treat a billing line as proof that a resource still exists or is unmanaged.
- [ ] **Enrich the Configuration:** retrieve current resource settings and dependencies through scoped cloud APIs, including networking, storage attachments, security settings, and identity references where supported. Record source evidence and unresolved attributes. Keep secret values out of generated code and use existing secret references or explicit placeholders.
- [ ] **Generate an Adoption Repository:** create supported Terraform/OpenTofu resource definitions, variables, outputs, provider constraints, dependency references, and import blocks using the actual provider import IDs. Preserve existing names and configuration; show unsupported resource types and missing dependencies for manual resolution. Offer modules after a faithful baseline is established.
- [ ] **Bootstrap the Management Workflow:** guide repository setup, environment boundaries, a protected remote state backend with locking and versioning, scoped CI identity, and plan/review steps for organizations that do not yet have an IaC pipeline.
- [ ] **Preview and Import in Stages:** show the resource-to-code mapping and planned imports for a selected application or environment. Require a reviewed adoption plan with zero resource creates, updates, replacements, or deletes before executing imports through the customer's approved workflow. Verify a subsequent clean plan and retain a state backup. Resolve configuration differences before adoption; optimize after the baseline is stable.
- [ ] **Track Adoption Coverage:** report discovered, eligible, generated, imported, verified, and unresolved resources with ownership and attributable spend. Keep unmatched billing charges visible. Prevent the same resource from being adopted into multiple states.

**First adoption deliverable shipped:** the four resource types above with a discovery preview, explicit gaps, generated import/configuration workflow, and a guarded path to a baseline. Continue the checklist for broader discovery, repository integration, managed-state reconciliation, dependency modeling, backend automation and coverage reporting. Full automatic reconstruction from billing data alone is not an acceptance criterion.

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
