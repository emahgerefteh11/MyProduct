import json
import re
import sys
import pathlib

def inject_data(html_path, json_path, output_path):
    print(f"Reading HTML from {html_path}...")
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print(f"Reading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()
        # Verify JSON is valid
        json.loads(json_content)

    print("Injecting data into HTML...")
    
    # We look for the init() function where the fetch happens
    # Pattern matches the try block start up to the reportData assignment
    # We want to replace the fetch logic with direct assignment
    
    # Target: 
    # async function init() {
    #    try {
    #        ... (fetch logic) ...
    #        reportData = await res.json();
    
    # Replacement:
    # async function init() {
    #    try {
    #        reportData = <JSON_CONTENT>;
    
    # Let's use a simpler regex that replaces the specific fetch block if possible,
    # or just injects the variable at the top of the script and modifies init to use it.
    
    # Strategy: Inject `const embeddedReportData = ...;` before `let reportData = null;`
    # And then change `init()` to use it.
    
    # 1. Inject Data Variable
    json_injection = f"const embeddedReportData = {json_content};\n"
    
    if "let reportData = null;" in html_content:
        html_content = html_content.replace(
            "let reportData = null;", 
            f"{json_injection}    let reportData = null;"
        )
    else:
        print("Warning: Could not find 'let reportData = null;' anchor.")

    # 2. Modify init() to use embedded data
    # We replace the fetch block with a simple assignment
    
    fetch_replacement = """
            // OFFLINE MODE: Use embedded data
            reportData = embeddedReportData;
            document.getElementById('last-updated').textContent = `Generated: ${new Date().toLocaleDateString()} (Offline)`;
            
            populateFilters();
            renderDashboard('all');
            return; // Skip fetch logic
            
            /* Original Fetch Logic (Disabled)
            const paths = ['report_data.json', 'MyProduct/report_data.json'];
    """
    
    if "const paths = ['report_data.json', 'MyProduct/report_data.json'];" in html_content:
        html_content = html_content.replace(
            "const paths = ['report_data.json', 'MyProduct/report_data.json'];", 
            fetch_replacement
        )
    else:
        # Fallback for V1 dashboard or other variations
        print("Warning: Could not find fetch logic to replace. Attempting generic injection.")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Success! Offline dashboard written to {output_path}")

if __name__ == "__main__":
    html_in = sys.argv[1] if len(sys.argv) > 1 else "myproduct/dashboard.html"
    json_in = sys.argv[2] if len(sys.argv) > 2 else "myproduct/report_data.json"
    html_out = sys.argv[3] if len(sys.argv) > 3 else "myproduct/dashboard.html" # Overwrite by default if 3rd arg provided, else be careful? 
    # Actually let's default to overwriting if run without args for simplicity in this flow
    
    inject_data(html_in, json_in, html_out)
