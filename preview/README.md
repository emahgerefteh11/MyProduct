# Design preview

Open [the preview](https://emahgerefteh11.github.io/MyProduct/preview/) or go
straight to the [sample dashboard](https://emahgerefteh11.github.io/MyProduct/preview/dashboard_demo.html).
The production experience remains at the site root.

This version explores clearer spend hierarchy, readable charts and tables,
a quieter navigation, a report-based homepage, and staged import/adoption flows.
It shares the existing billing, chart, and Terraform engines. Account connections
remain planned; local discovery and review remain required for adoption.

Accepted preview reports use the separate IndexedDB database
`cloud-cost-atlas-design-preview`. Importing, replacing, or clearing a preview
report does not change a report stored by the production pages. Sample reports
remain separate from imported reports.

Run `python build_preview.py` from the repository root to regenerate the preview
dashboard, demo, upload/adoption pages, and adapted UI/worker/storage scripts.
Edit `preview/index.html`, `preview/preview.css`, and `preview/preview.js` directly.
Generated preview pages use the root CSS and analysis libraries and should be
served from their `preview/` path rather than copied alone.

To check locally, serve the repository over localhost and run
`python tests/browser_preview.py` with Playwright Chromium installed. Set
`ATLAS_CHROMIUM_PATH` when using a system Chromium installation.
