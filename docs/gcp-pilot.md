# GCP billing pilot: first local acceptance

GCP fixture work can proceed while access to a live AWS account is pending. The prototype uses an injected reader and an in-memory accepted report per tenant, source and invoice month. No real BigQuery query, cloud account, role grant, cloud spend, hosted connection or authentication is included.

## Source and scope

Start with a [BigQuery Standard usage cost export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/standard-usage) from one approved billing account and one explicit `invoice.month` (not the calendar month of `usage_start_time`). Compare the same net cost column on both the upload and connector paths. Google reports credits as a repeated field. Add the signed credits to each row exactly once without duplicating the original charge across credit rows. Retain tax, adjustment and rounding rows unless an explicit finance rule requires otherwise; separate billed-invoice reconciliation from matching two Atlas input paths.

The local adapter expects one UTF-8 CSV projection with headers `invoice.month, Cost, currency, service.description, AtlasGcpCostBasis`; optional Product/Platform/Environment use existing explicit importer mappings. `Cost` must be pre-netted; set `AtlasGcpCostBasis` to `net_with_credits` only when that is true. Use the same bytes for manual upload. Both routes retain the same explicit invoice-month metadata for single-month exports. The manual uploader can still produce an aggregate from mixed, missing or invalid months, but does not label it as an invoice-month snapshot. Usage coverage is recorded separately. A raw Standard export with a separate credits field is not this projection. The reader supplies an ordered positive integer revision for the scope; this is a local test contract, not a native BigQuery export version. A hosted reader still needs a stable snapshot and verified identity. Do not use an arbitrary per-row export timestamp as if it were a complete revision.

## Native-shaped export rehearsal

`local_gcp_projection.cjs` projects one complete, decoded Standard-export-shaped snapshot to the same CSV contract. It requires the exact billed account and invoice month on every row, with one explicit currency. An empty `credits` array means none; missing or malformed credits fail instead of silently raising totals. Signed credit amounts are added to each original cost once using decimal arithmetic. Rows are not deduplicated or filtered by usage dates or cost type. Keep charge, tax, adjustment and rounding rows in scope. The resulting ordering is deterministic for identical content, but is not proof that a live export is complete or an indication of revision order. The optional `atlasProduct` field is an already-resolved internal ownership label, not a native Standard export field; unmapped rows stay unallocated.

The browser accepts this flattened projection, not raw nested BigQuery result JSON. No BigQuery API or live export was used.

## Verified with sample data

- $100 net: $70 Payments and $30 unallocated through both paths.
- Exact retry stays at $100 with the original acceptance time. A newer revision replaces it at $105 with $35 unallocated.
- Wrong month, mixed currencies and non-net inputs are rejected; a changed or older revision cannot replace the accepted result.
- Native-shaped samples include two credits on one charge, uncredited rows, tax, adjustment and rounding, plus September usage on the October invoice. Upload and connector agree at $100 on first pass and reordered retry, then $105 after correction. Wrong account or invalid credits leave the accepted report intact.

## Compare saved CSV invoice months

1. On the [Import billing page](https://emahgerefteh11.github.io/MyProduct/import.html), upload one invoice month at a time. You may select multiple non-overlapping parts of that month. Every row needs the same valid `invoice.month` and an explicit billing currency. Earlier usage dates do not change the invoice month.
2. In the validated preview, review warnings and select **Save invoice month**. Use a consistent label for the same billing account or export scope; Atlas does not verify that manually entered scopes are equivalent. For an unmarked cost column, say whether it includes credits only if you know. Do not call a source “after credits” if the mapped amount is gross and credits are in a separate column.
3. In **Compare Google Cloud invoices**, choose two months. The difference is only shown for the same named scope, currency, cost basis, and known credit treatment. Unknown or conflicting credit treatment still allows viewing totals, but not a delta.
4. Re-import a correction under the same scope and confirm the replacement. A same-total retry changes nothing. A correction never adds to the previous total or changes another month. Remove an incorrect saved summary before replacing it with an incompatible currency or cost treatment.

Only invoice totals and comparison labels are saved, in this browser on this site. Original CSVs and report details are not kept in history. Opening or clearing the current dashboard report does not change your saved months. The design preview keeps separate history.

## Verify a completed result before accepting a revision

The local `collectGcpQueryProjection` rehearsal expects a successful terminal BigQuery job, then pages from the same job project, ID and location. It follows the returned continuation chain exactly once and checks that the retrieved row count matches the declared total before handing any bytes to the connector. A truncated, empty, failed, oversized or inconsistent result leaves the prior accepted report unchanged. A later valid, complete result can still replace it. This validates result delivery, not financial completeness of the source: use the scoped query and reconcile to the approved invoice separately.

A real reader needs allowlisted account/table scope, authenticated job status, bounded retries and raw-response byte limits before decoding. Estimates should come from a [BigQuery dry run](https://docs.cloud.google.com/bigquery/docs/running-queries#dry-run); a live on-demand query also needs an expressly approved byte ceiling. [BigQuery result paging](https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/jobs/getQueryResults) is tied to the original job, not a fresh query per page. A job ID or a result's page count does not order billing corrections; keep accepted revision order separate. Do not treat [BigQuery time travel](https://docs.cloud.google.com/bigquery/docs/access-historical-data) as permanent invoice history or use it by default for an identity restricted by row-level policy.

## One-invoice live pilot checklist (not yet run)

1. **Confirm scope before requesting credentials or changing access.** Record a pilot alias, one existing Standard export table, its dataset location, the billing account scope, one finalized invoice month (`YYYYMM`), and the project that would pay for the query. Keep actual identifiers and invoice evidence in the approved private workspace, never in this repository or the public demo. Use a read-only identity scoped to the approved table or authorized view and grant job execution only in the approved query project, if separately authorized. A SQL account filter does not grant or enforce access control.
2. **Verify source coverage.** Check when export was enabled and last delivered, whether the location is single-region or multi-region, and whether the table or dataset was switched. Google documents that a first US/EU multi-region export can backfill to the start of the previous month and take up to five days to catch up; a regional export starts at enablement. Do not assume a re-enabled or moved export backfills a gap. If the selected invoice began before available coverage, or delivery is still catching up, mark the pilot blocked or provisional rather than reporting a reconciliation failure. [Cloud Billing export timing and limits](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-setup#limitations)
3. **Estimate the exact query, then get approval for live execution.** Once scoped access is approved, use a dry run for the same SQL, table, account, month, project and location intended for the real query. Record the estimated bytes and estimate uncertainty. A `LIMIT` isn't a billing cap, and a zero-byte estimate from a target protected by row-level access policies is not evidence that the scan is free. For an on-demand run get approval for a precise `maximumBytesBilled` and use that setting; stop if the estimate cannot be verified or exceeds the ceiling. No dataset creation, permanent result storage, capacity reservation, or scheduled refresh as part of this pilot. [BigQuery estimate and cost controls](https://docs.cloud.google.com/bigquery/docs/best-practices-costs#restrict_the_number_of_bytes_billed_per_query)
4. **Reconcile one completed job.** Read all result pages from the same job reference, verify the reported row count, and project the retrieved rows once. Use those exact CSV bytes in upload and connector parity and compare source SHA-256, total and allocation. Separately compare the selected invoice's finalized net amounts on the same currency, account/scope and cost treatment. Record each disposition separately: source coverage, query and pages complete, upload/connector agree, approved invoice reconciles. Neither a successful job nor matched upload bytes proves the source includes the finalized invoice.
5. **Test failures without changing accepted data.** A denied scope or incomplete read must leave the prior accepted report untouched. Replay only the already accepted immutable result for an exact retry; a refreshed query is a new observation and not a revision number by itself. Keep hosted authorization, revocation and tenant isolation as separate live release checks.

### Candidate GoogleSQL for the dry run

Use an allowlisted Standard table or an approved view that exposes these fields. Replace the illustrative table name under review, not from user-entered UI text; bind the account and month as named `STRING` parameters. Preserve original cost and individual signed credits because the Atlas projection applies credits once. No usage-month filter, cost-type filter, aggregation, or `LIMIT`. This query has not been executed against BigQuery.

```sql
SELECT
  src.billing_account_id,
  STRUCT(src.invoice.month AS month) AS invoice,
  src.currency,
  STRUCT(src.service.description AS description) AS service,
  CAST(src.cost AS STRING) AS cost,
  ARRAY(
    SELECT AS STRUCT CAST(credit.amount AS STRING) AS amount
    FROM UNNEST(src.credits) AS credit
  ) AS credits,
  src.cost_type,
  FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E6SZ', src.usage_start_time, 'UTC') AS usage_start_time,
  FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E6SZ', src.usage_end_time, 'UTC') AS usage_end_time
FROM `approved-project.approved_dataset.gcp_billing_export_v1_approved` AS src
WHERE src.billing_account_id = @billing_account
  AND src.invoice.month = @invoice_month
```

## Next

An owner must identify the existing export and approve scoped read access before attempting the dry run. Keep finance reconciliation and live authorization separate from the local parity checks. Browser-local history does not sync between devices or accept connector revisions automatically. Live BigQuery execution, cloud spend, and hosted deployment require approval.
