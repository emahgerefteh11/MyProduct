"""
generate_dashboard.py — Build the demo dashboard.

Reads dashboard.html (the live version) and injects report_data.json directly
into it as a JS constant, producing dashboard_demo.html. The dashboard's init()
function auto-detects the embedded data and enters demo mode automatically.

Usage:
    python generate_dashboard.py                              # uses defaults
    python generate_dashboard.py <html_in> <json_in> <html_out>
"""

import json
import sys

def inject_data(html_path, json_path, output_path):
    print(f"Reading HTML from {html_path}...")
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print(f"Reading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()
        json.loads(json_content)  # validate

    print("Injecting data into HTML...")

    # Inject embeddedReportData before the reportData variable declaration.
    # The dashboard's init() checks `typeof embeddedReportData !== 'undefined'`
    # and uses it automatically — no other changes needed.
    anchor = "let reportData = null;"
    injection = f"const embeddedReportData = {json_content};\n    "

    if anchor in html_content:
        html_content = html_content.replace(anchor, injection + anchor, 1)
    else:
        print("Warning: Could not find anchor 'let reportData = null;' — data not injected.")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Success! Demo dashboard written to {output_path}")

if __name__ == "__main__":
    html_in  = sys.argv[1] if len(sys.argv) > 1 else "dashboard.html"
    json_in  = sys.argv[2] if len(sys.argv) > 2 else "report_data.json"
    html_out = sys.argv[3] if len(sys.argv) > 3 else "dashboard_demo.html"

    inject_data(html_in, json_in, html_out)
