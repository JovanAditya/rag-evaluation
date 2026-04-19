"""
Generate Report - Visualisasi Hasil Evaluasi RAG.

Membuat laporan perbandingan lengkap dengan tabel dan grafik
untuk lampiran skripsi.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


SCRIPT_DIR = Path(__file__).resolve().parent

# Color palette
COLORS = {
    "baseline": "#3498db",  # Blue
    "hybrid": "#e74c3c",    # Red
    "bg": "#f8f9fa",
    "text": "#2c3e50",
    "grid": "#ecf0f1",
}


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_time_comparison(summary: Dict, output_dir: str):
    """Bar chart: perbandingan waktu respons."""
    if not HAS_MATPLOTLIB:
        return

    pipelines = list(summary.keys())
    metrics = ["avg_retrieval_time", "avg_generation_time"]
    labels = ["Retrieval Time (s)", "Generation Time (s)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    width = 0.3

    for i, pipeline in enumerate(pipelines):
        values = [summary[pipeline].get(m, 0) for m in metrics]
        color = COLORS.get(pipeline, COLORS["baseline"])
        bars = ax.bar(
            [xi + i * width for xi in x], values, width,
            label=pipeline.capitalize(), color=color, alpha=0.85,
            edgecolor="white", linewidth=1.5
        )
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_ylabel("Waktu (detik)", fontsize=12)
    ax.set_title("Perbandingan Waktu Respons: Baseline vs Hybrid", fontsize=14, fontweight="bold")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor(COLORS["bg"])
    fig.tight_layout()

    filepath = os.path.join(output_dir, "time_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {filepath}")


def plot_quality_comparison(summary: Dict, output_dir: str):
    """Bar chart: perbandingan kualitas."""
    if not HAS_MATPLOTLIB:
        return

    pipelines = list(summary.keys())
    metrics = ["avg_relevance_score", "avg_confidence"]
    labels = ["Avg Relevance Score", "Avg Confidence"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    width = 0.3

    for i, pipeline in enumerate(pipelines):
        values = [summary[pipeline].get(m, 0) for m in metrics]
        color = COLORS.get(pipeline, COLORS["baseline"])
        bars = ax.bar(
            [xi + i * width for xi in x], values, width,
            label=pipeline.capitalize(), color=color, alpha=0.85,
            edgecolor="white", linewidth=1.5
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Perbandingan Kualitas Retrieval: Baseline vs Hybrid", fontsize=14, fontweight="bold")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor(COLORS["bg"])
    
    # Calculate proper ylim based on max value to prevent text cutoff
    max_val = max([max([summary[p].get(m, 0) for m in metrics]) for p in pipelines] + [1.0])
    ax.set_ylim(0, max_val * 1.15)  # Add 15% headroom for text labels
    
    fig.tight_layout()

    filepath = os.path.join(output_dir, "quality_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {filepath}")


def plot_radar_chart(summary: Dict, output_dir: str):
    """Radar chart: perbandingan multi-dimensi."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    categories = [
        "Relevance\nScore", "Confidence", "Speed\n(inv. time)",
        "Context\nLength", "Docs\nRetrieved"
    ]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    for pipeline_name, metrics in summary.items():
        # Normalize values to 0-1 range
        max_time = max(m.get("avg_total_time", 1) for m in summary.values())
        speed = 1 - (metrics.get("avg_total_time", 0) / max_time) if max_time > 0 else 0

        max_ctx = max(m.get("avg_context_length", 1) for m in summary.values())
        ctx_norm = metrics.get("avg_context_length", 0) / max_ctx if max_ctx > 0 else 0

        max_docs = max(m.get("avg_documents_retrieved", 1) for m in summary.values())
        docs_norm = metrics.get("avg_documents_retrieved", 0) / max_docs if max_docs > 0 else 0

        values = [
            metrics.get("avg_relevance_score", 0),
            metrics.get("avg_confidence", 0),
            speed,
            ctx_norm,
            docs_norm,
        ]
        values += values[:1]

        color = COLORS.get(pipeline_name, COLORS["baseline"])
        ax.plot(angles, values, "o-", linewidth=2, label=pipeline_name.capitalize(), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart: Baseline vs Hybrid", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=11)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "radar_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {filepath}")


def plot_per_question_comparison(results: List[Dict], output_dir: str):
    """Bar chart: total time per pertanyaan."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    # Get unique questions
    q_ids = sorted(set(r["question_id"] for r in results))

    # Average times per question per pipeline
    pipelines = sorted(set(r["pipeline"] for r in results))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(q_ids))
    width = 0.35

    for i, pipeline in enumerate(pipelines):
        times = []
        for q_id in q_ids:
            q_results = [
                r["metrics"]["total_time"]
                for r in results
                if r["question_id"] == q_id and r["pipeline"] == pipeline and r.get("success")
            ]
            times.append(sum(q_results) / len(q_results) if q_results else 0)

        color = COLORS.get(pipeline, COLORS["baseline"])
        ax.bar(x + i * width, times, width, label=pipeline.capitalize(), color=color, alpha=0.85)

    ax.set_xlabel("ID Pertanyaan", fontsize=11)
    ax.set_ylabel("Waktu Total (detik)", fontsize=11)
    ax.set_title("Waktu Respons per Pertanyaan", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor(COLORS["bg"])
    fig.tight_layout()

    filepath = os.path.join(output_dir, "per_question_time.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {filepath}")


def generate_html_report(results: Dict, output_dir: str):
    """Generate HTML report."""
    summary = results.get("summary", {})
    metadata = results.get("metadata", {})
    pipelines = list(summary.keys())

    # Check for images
    images = []
    for img_name in ["time_comparison.png", "quality_comparison.png",
                      "radar_comparison.png", "per_question_time.png"]:
        if os.path.exists(os.path.join(output_dir, img_name)):
            images.append(img_name)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Evaluasi RAG</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 40px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ font-size: 28px; margin-bottom: 10px; color: #2c3e50; }}
        h2 {{ font-size: 20px; margin: 30px 0 15px; color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .meta {{ color: #7f8c8d; margin-bottom: 30px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #2c3e50; color: white; padding: 12px 16px; text-align: left; font-weight: 600; }}
        td {{ padding: 10px 16px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .highlight {{ background: #e8f8e8; font-weight: bold; }}
        img {{ max-width: 100%; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #95a5a6; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Laporan Evaluasi RAG Akademik</h1>
    <p class="meta">
        Eksperimen: {metadata.get('experiment_name', '-')}<br>
        Tanggal: {metadata.get('timestamp', '-')}<br>
        Pertanyaan: {metadata.get('num_questions', 0)} | Pengulangan: {metadata.get('num_runs', 1)}
    </p>

    <h2>Ringkasan Perbandingan</h2>
    <table>
        <thead>
            <tr>
                <th>Metrik</th>
                {''.join(f'<th>{p.capitalize()}</th>' for p in pipelines)}
            </tr>
        </thead>
        <tbody>"""

    metric_labels = {
        "success_rate": ("Success Rate", lambda v: f"{v:.1%}"),
        "avg_mrr": ("Mean Reciprocal Rank (MRR)", lambda v: f"{v:.4f}"),
        "avg_precision_at_k": ("Precision@K", lambda v: f"{v:.4f}"),
        "avg_recall_at_k": ("Recall@K", lambda v: f"{v:.4f}"),
        "avg_faithfulness": ("Faithfulness (RAGAS)", lambda v: f"{v:.4f}"),
        "avg_answer_relevancy": ("Answer Relevancy (RAGAS)", lambda v: f"{v:.4f}"),
        "avg_context_precision": ("Context Precision (RAGAS)", lambda v: f"{v:.4f}"),
        "avg_relevance_score": ("Avg Relevance Score (Raw)", lambda v: f"{v:.4f}"),
        "avg_confidence": ("Avg Confidence", lambda v: f"{v:.4f}"),
        "avg_documents_retrieved": ("Avg Documents Retrieved", lambda v: f"{v:.1f}"),
        "avg_retrieval_time": ("Avg Retrieval Time (s)", lambda v: f"{v:.4f}"),
        "avg_generation_time": ("Avg Generation Time (s)", lambda v: f"{v:.4f}"),
        "avg_total_time": ("Avg Total Time (s)", lambda v: f"{v:.4f}"),
        "avg_answer_length": ("Avg Answer Length (chars)", lambda v: f"{v:.0f}"),
        "avg_context_length": ("Avg Context Length", lambda v: f"{v:.0f}"),
    }

    for key, (label, fmt) in metric_labels.items():
        values = [summary[p].get(key, 0) for p in pipelines]
        best_idx = values.index(max(values)) if "time" not in key else values.index(min(values))

        html += f"\n            <tr><td><strong>{label}</strong></td>"
        for i, p in enumerate(pipelines):
            cls = ' class="highlight"' if i == best_idx else ""
            html += f'<td{cls}>{fmt(values[i])}</td>'
        html += "</tr>"

    html += """
        </tbody>
    </table>

    <h2>Visualisasi</h2>
    <div class="grid">"""

    for img in images:
        html += f'\n        <div><img src="{img}" alt="{img}"></div>'

    html += f"""
    </div>

    <div class="footer">
        <p>Generated by RAG Evaluation Tool — Universitas Mercu Buana</p>
    </div>
</div>
</body>
</html>"""

    filepath = os.path.join(output_dir, "report.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"  ✓ {filepath}")
    return filepath


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate laporan evaluasi RAG")
    parser.add_argument(
        "--input", required=True, help="Path ke file hasil evaluasi JSON"
    )
    parser.add_argument(
        "--output", default=None, help="Direktori output (default: sama dengan input)"
    )
    args = parser.parse_args()

    # Load results
    results = load_results(args.input)
    
    # Generate output directory name (default: subfolder based on input filename)
    if args.output:
        output_dir = args.output
    else:
        filename_without_ext = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = os.path.join(os.path.dirname(args.input), f"{filename_without_ext}_report")
        
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating report...")
    summary = results.get("summary", {})
    all_results = results.get("results", [])

    # Generate plots
    if HAS_MATPLOTLIB:
        logger.info("Generating charts...")
        plot_time_comparison(summary, output_dir)
        plot_quality_comparison(summary, output_dir)
        plot_radar_chart(summary, output_dir)
        plot_per_question_comparison(all_results, output_dir)
    else:
        logger.warning("Skipping charts (matplotlib not installed)")

    # Generate HTML report
    logger.info("Generating HTML report...")
    report_path = generate_html_report(results, output_dir)

    print(f"\n✅ Laporan berhasil dibuat!")
    print(f"   HTML: {report_path}")
    if HAS_MATPLOTLIB:
        print(f"   Charts: {output_dir}/")


if __name__ == "__main__":
    main()
