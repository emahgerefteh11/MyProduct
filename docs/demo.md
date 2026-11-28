# Cloud Cost Atlas demo: no AWS setup required

**Two-minute version: show who owns the spend.**

1. Open the [sample dashboard](https://emahgerefteh11.github.io/MyProduct/dashboard_demo.html). Point out the **DEMO DATA** label.
2. In the platform chart, select **PlatformE**. Open **Product costs** to show which products drive the selected platform's spend. Remove the platform filter to get back to the full sample.
3. Say: “This is sample data. The demo answers where the spend went and which product owns it. Next I can show the same analysis on a CSV.”

**Five-minute version: show a bill and its correction.**

Before presenting, download the [original synthetic AWS CSV](../tests/fixtures/cur/synthetic.csv) and the [corrected CSV](demo/aws-corrected.csv). Use a fresh browser profile or window with separate storage, so you do not replace a real imported report. If downloads or browser storage are unavailable, stay on the sample dashboard.

1. Open [Import billing reports](https://emahgerefteh11.github.io/MyProduct/import.html). Choose **only** the original file, select **Review mappings**, and leave the detected AWS mapping and currency as-is. Select **Validate & preview**, acknowledge the review, then select **Open dashboard**.
2. The total is **$100**. Payments owns **$70** ($80 of storage less a $10 credit); **$30** of compute has no product tag and stays unallocated. Open **Report actions → Download report JSON** if someone wants to inspect the allocation. The import stays in that browser, not in a hosted AWS account.
3. To show a correction, choose **Report actions → Import a new report** and import **only** the corrected file. It replaces the saved browser report: **$105** total, **$70** Payments and **$35** unallocated. Do not select both files together; they are revisions of the same bill, not two independent bills.
4. After the demo, clear the sample import via **Report actions → Clear imported report**, or close the disposable browser profile.

**What isn't live yet:** These steps show sample and manual upload. The AWS connector retry, revision and disconnect behavior has passed synthetic checks, but the public demo is not connected to AWS. A private deployment with an approved AWS export is a separate pilot.

## Optional GCP walkthrough: compare two months without a cloud account

Download [October](demo/gcp-october.csv), [November](demo/gcp-november.csv), and the [October correction](demo/gcp-october-corrected.csv). Use synthetic CSVs and a separate browser profile for this demo.

1. [Import billing](https://emahgerefteh11.github.io/MyProduct/import.html) and choose October alone. Validate, acknowledge the warnings, and **Save invoice month** under `Atlas demo`. October stays October even though one usage row is from September.
2. Reload the import page and repeat for November under `Atlas demo`. In **Compare Google Cloud invoices**, show **$100 in October** next to **$130 in November**, a **+$30** difference.
3. Repeat with the October correction and confirm that it replaces October. The comparison now shows **$105 versus $130**, or **+$25**. Refresh to show both totals are still there; the current dashboard report was not changed by saving the history.
4. Remove the two demo months afterward, or close the separate temporary browser profile. This walkthrough does not connect to BigQuery.
