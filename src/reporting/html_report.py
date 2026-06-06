from __future__ import annotations

import json
from pathlib import Path


def write_report(method: str, rows: int, seed: int, metrics: dict, report_path: Path, run_name: str) -> None:
    """Write a simple standalone HTML report for a synthetic-data run."""
    base = f"../outputs/{run_name}/plots"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Synthetic Data Artist - Report</title>
<style>body{{font-family:Arial;margin:24px}} pre{{background:#f6f8fa;padding:12px}}</style></head>
<body>
<h1>Synthetic Data Artist</h1>
<p>Method: <b>{method}</b> · Rows: {rows} · Seed: {seed}</p>
<h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre>
<h2>Distribution Overlap</h2><img src="{base}/distribution_overlap.png">
<h2>PCA Projection</h2><img src="{base}/pca_projection.png">
<h2>Correlation</h2><img src="{base}/correlation_heatmap.png">
<h2>Pairplot</h2><img src="{base}/pairplot_comparison.png">
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
