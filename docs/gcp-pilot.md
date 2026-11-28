# GCP billing pilot: first local acceptance

GCP fixture work can proceed while access to a live AWS account is pending. The prototype uses an injected reader and an in-memory accepted report per tenant, source and invoice month. No real BigQuery query, cloud account, role grant, cloud spend, hosted connection or authentication is included.

## Source and scope

Start with a [BigQuery Standard usage cost export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/standard-usage) from one approved billing account and one explicit `invoice.month` (not the calendar month of `usage_start_time`). Compare the same net cost column on both the upload and connector paths. Google reports credits as a repeated field. Add the signed credits to each row exactly once without duplicating the original charge across credit rows. Retain tax, adjustment and rounding rows unless an explicit finance rule requires otherwise; separate billed-invoice reconciliation from matching two Atlas input paths.

The local adapter expects one UTF-8 CSV projection with headers `invoice.month, Cost, currency, service.description, AtlasGcpCostBasis`; optional Product/Platform/Environment use existing explicit importer mappings. `Cost` must be pre-netted; set `AtlasGcpCostBasis` to `net_with_credits` only when that is true. Use the same bytes for manual upload. Both routes retain the same explicit invoice-month metadata for single-month exports. The manual uploader can still produce an aggregate from mixed, missing or invalid months, but does not label it as an invoice-month snapshot. Usage coverage is recorded separately. A raw Standard export with a separate credits field is not this projection. The reader supplies an ordered positive integer revision for the scope; this is a local test contract, not a native BigQuery export version. A hosted reader still needs a stable snapshot and verified identity. Do not use an arbitrary per-row export timestamp as if it were a complete revision.

## Verified with sample data

- $100 net: $70 Payments and $30 unallocated through both paths.
- Exact retry stays at $100 with the original acceptance time. A newer revision replaces it at $105 with $35 unallocated.
- Wrong month, mixed currencies and non-net inputs are rejected; a changed or older revision cannot replace the accepted result.

## Compare saved CSV invoice months

1. On the [Import billing page](https://emahgerefteh11.github.io/MyProduct/import.html), upload one invoice month at a time. You may select multiple non-overlapping parts of that month. Every row needs the same valid `invoice.month` and an explicit billing currency. Earlier usage dates do not change the invoice month.
2. In the validated preview, review warnings and select **Save invoice month**. Use a consistent label for the same billing account or export scope; Atlas does not verify that manually entered scopes are equivalent. For an unmarked cost column, say whether it includes credits only if you know. Do not call a source “after credits” if the mapped amount is gross and credits are in a separate column.
3. In **Compare Google Cloud invoices**, choose two months. The difference is only shown for the same named scope, currency, cost basis, and known credit treatment. Unknown or conflicting credit treatment still allows viewing totals, but not a delta.
4. Re-import a correction under the same scope and confirm the replacement. A same-total retry changes nothing. A correction never adds to the previous total or changes another month. Remove an incorrect saved summary before replacing it with an incompatible currency or cost treatment.

Only invoice totals and comparison labels are saved, in this browser on this site. Original CSVs and report details are not kept in history. Opening or clearing the current dashboard report does not change your saved months. The design preview keeps separate history.

## Next

Test the net projection itself with synthetic Standard export rows, including late-monetized usage, multiple signed credits and charges with no credits. Then validate a real approved export and authenticated BigQuery reader. This browser-local history does not sync between devices or accept connector revisions automatically. Live BigQuery access, query spend and deployment require approval.
