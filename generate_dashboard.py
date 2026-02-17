import json
import datetime
import sys

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Infrastructure Cost Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f4f6f8; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #2c3e50; margin-top: 0; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; flex: 1; min-width: 300px; }}
        .row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
        .stat {{ font-size: 2.5em; font-weight: bold; color: #27ae60; text-align: center; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; text-align: center; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; font-weight: 600; color: #555; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .badge {{ padding: 6px 12px; border-radius: 4px; font-size: 0.85em; font-weight: 600; background: #e0e0e0; color: #333; margin-right: 5px; margin-bottom: 5px; display: inline-block; }}
        .chart-container {{ position: relative; height: 300px; width: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="card">
                <h1>Infrastructure Dashboard</h1>
                <p>Generated: {generated_at}</p>
                <p>Source: Local Report Data</p>
            </div>
            <div class="card">
                <div class="stat-label">Total Monthly Cost</div>
                <div class="stat">${total_cost}</div>
            </div>
        </div>

        <div class="row">
            <div class="card">
                <h2>Cost by Provider</h2>
                <div class="chart-container">
                    <canvas id="providerChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Cost by Platform</h2>
                <div class="chart-container">
                    <canvas id="platformChart"></canvas>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Product Breakdown</h2>
            <table id="productTable">
                <thead>
                    <tr>
                        <th>Product</th>
                        <th>Cost</th>
                        <th>% of Total</th>
                        <th>Service Count</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Rows injected via JS -->
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Terraform Scope</h2>
            <div id="tfStatus"></div>
        </div>
    </div>

    <script>
        const reportData = {json_data};

        // Format currency helper
        const formatCurrency = (val) => {{
            return new Intl.NumberFormat('en-US', {{ style: 'currency', currency: 'USD' }}).format(val);
        }};

        // Render Charts
        if (reportData.providers && reportData.providers.length > 0) {{
            const ctxProvider = document.getElementById('providerChart').getContext('2d');
            new Chart(ctxProvider, {{
                type: 'doughnut',
                data: {{
                    labels: reportData.providers.map(p => p.provider.toUpperCase()),
                    datasets: [{{
                        data: reportData.providers.map(p => p.cost),
                        backgroundColor: ['#FF9900', '#0078D4', '#4285F4', '#7f8c8d']
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
        }}

        if (reportData.platforms && reportData.platforms.length > 0) {{
            const ctxPlatform = document.getElementById('platformChart').getContext('2d');
            new Chart(ctxPlatform, {{
                type: 'bar',
                data: {{
                    labels: reportData.platforms.map(p => p.platform),
                    datasets: [{{
                        label: 'Cost',
                        data: reportData.platforms.map(p => p.cost),
                        backgroundColor: '#3498db'
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
        }}

        // Render Product Table
        const tbody = document.querySelector('#productTable tbody');
        if (reportData.products) {{
            reportData.products.forEach(p => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${{p.product}}</strong></td>
                    <td>${{formatCurrency(p.cost)}}</td>
                    <td>${{p.pctOfTotal}}%</td>
                    <td>${{p.serviceCount}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Render Terraform Scope Badges
        const tfDiv = document.getElementById('tfStatus');
        if (reportData.platforms) {{
            reportData.platforms.forEach(plat => {{
                if (plat.products) {{
                    plat.products.forEach(prod => {{
                        const badge = document.createElement('span');
                        badge.className = 'badge';
                        badge.textContent = `${{plat.platform}} / ${{prod.product}}`;
                        tfDiv.appendChild(badge);
                    }});
                }}
            }});
        }}
    </script>
</body>
</html>
"""

def generate_dashboard(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Pre-format total cost for HTML injection
    total_cost_str = "{:,.2f}".format(data.get("totalCost", 0.0))
    
    # Create the HTML content
    # We pass the raw JSON string into the JS variable `reportData`
    html_content = HTML_TEMPLATE.format(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_cost=total_cost_str,
        json_data=json.dumps(data)
    )

    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated dashboard at {output_path}")

if __name__ == "__main__":
    json_input = sys.argv[1] if len(sys.argv) > 1 else "myproduct/report_data.json"
    html_output = sys.argv[2] if len(sys.argv) > 2 else "myproduct/dashboard.html"
    generate_dashboard(json_input, html_output)
