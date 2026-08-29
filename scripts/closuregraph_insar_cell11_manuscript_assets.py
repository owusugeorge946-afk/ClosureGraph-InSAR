"""Cell 11: manuscript-quality figures and tables for ClosureGraph-InSAR.

Run this only AFTER Cell 10 has completed:
    %run -i /content/closuregraph_insar_cell11_manuscript_assets.py

It never trains models and never opens the locked-test shards.  It only reads
the frozen CSV/JSON outputs already written by Cells 8, 9R, and 10, then saves
publication-ready PNG (900 dpi), PDF, SVG, CSV, LaTeX, and caption files.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
from datetime import datetime, timezone

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path("/content/drive/MyDrive/ClosureGraph_InSAR")
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
VERSION = "manuscript_assets_v3"
OUT = RESULTS / VERSION
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
OUT_LOG = OUT / "logs"

for folder in (OUT_FIG, OUT_TAB, OUT_LOG):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.titlesize": 14, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def banner(text: str) -> None:
    print("=" * 96)
    print(text)
    print("=" * 96)


def locate_exact(name: str) -> Path:
    matches = list(TABLES.glob(name)) + list(LOGS.glob(name)) + list(FIGURES.glob(name))
    if not matches:
        raise FileNotFoundError(f"Required Cell-10 output is missing: {name}")
    return matches[0]


def optional_first(*patterns: str) -> Path | None:
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def column(df: pd.DataFrame, *needles: str) -> str | None:
    """Find a column by exact or case-insensitive substring match."""
    normalized = {str(c).lower(): c for c in df.columns}
    for needle in needles:
        if needle.lower() in normalized:
            return normalized[needle.lower()]
    for needle in needles:
        needle = needle.lower()
        for low, original in normalized.items():
            if needle in low:
                return original
    return None


def number_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for suffix, kwargs in ((".png", {"dpi": 900}), (".pdf", {}), (".svg", {})):
        path = OUT_FIG / f"{stem}{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        paths.append(str(path))
    plt.show()
    plt.close(fig)
    return paths


def save_table(df: pd.DataFrame, stem: str, index: bool = False) -> list[str]:
    csv_path = OUT_TAB / f"{stem}.csv"
    tex_path = OUT_TAB / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    try:
        df.to_latex(tex_path, index=index, escape=False, float_format=lambda x: f"{x:.4f}")
    except Exception as exc:
        tex_path.write_text(f"% LaTeX export unavailable: {exc}\n", encoding="utf-8")
    return [str(csv_path), str(tex_path)]


def model_label(value: object) -> str:
    raw = str(value).lower().replace("_", " ")
    if "amplitude" in raw or "promoted" in raw:
        return "Graph + amplitude"
    if "baseline" in raw:
        return "Baseline"
    return str(value).replace("_", " ").title()


def metric_direction(name: str) -> bool:
    """True when lower is better."""
    text = name.lower()
    return any(token in text for token in ("mae", "rmse", "mse", "bias", "loss", "error"))


banner("CELL 11: CLOSUREGRAPH-INSAR MANUSCRIPT ASSET PACKAGE")
print("This cell reads saved results only: no training and no locked-test shard access.")
print(f"Project root : {ROOT}")

# Cell 10 completion is the guard that prevents Cell 11 from running early.
complete_path = locate_exact("closuregraph_cell10_locked_test_complete_locked_test_v1.json")
summary_path = locate_exact("closuregraph_cell10_locked_test_summary_locked_test_v1.csv")
seed_path = locate_exact("closuregraph_cell10_locked_test_seed_summary_locked_test_v1.csv")
sig_path = locate_exact("closuregraph_cell10_locked_test_significance_locked_test_v1.csv")
records_path = locate_exact("closuregraph_cell10_locked_test_records_locked_test_v1.csv")

with complete_path.open("r", encoding="utf-8") as fh:
    completion = json.load(fh)
summary = pd.read_csv(summary_path)
seed_summary = pd.read_csv(seed_path)
significance = pd.read_csv(sig_path)
records = pd.read_csv(records_path)

print("Cell-10 completion audit: FOUND")
print(f"Locked-test records      : {len(records):,}")

produced: dict[str, list[str]] = {"figures": [], "tables": []}

# -------------------------------------------------------------------------
# FIGURE 1 — protocol schematic
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14.4, 3.25))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 3.25)
ax.axis("off")
boxes = [
    (0.35, 1.02, 2.75, 0.92, "1. Production dataset\n20,000 train | 2,000 validation\n20 acquisitions × 128 × 128", "#d8eaf8"),
    (3.82, 1.02, 2.75, 0.92, "2. Model development\nBaseline and Graph + amplitude\nThree independent training seeds", "#ececec"),
    (7.29, 1.02, 2.75, 0.92, "3. Checkpoint freeze\nValidation common score only\nNo locked-test feedback", "#dff1e5"),
    (10.76, 1.02, 2.75, 0.92, "4. Final evaluation\n2,000 locked-test sequences\nOpened exactly once", "#fce8cc"),
]
for x, y, w, h, text, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                linewidth=1.2, edgecolor="#4c4c4c", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.1, weight="bold", linespacing=1.25)
for start, end in ((3.15, 3.75), (6.62, 7.22), (10.09, 10.69)):
    ax.add_patch(FancyArrowPatch((start, 1.48), (end, 1.48), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.5, color="#555555"))
ax.plot([0.65, 10.05], [0.54, 0.54], color="#55728c", lw=1.2)
ax.plot([10.88, 13.27], [0.54, 0.54], color="#a86e25", lw=1.2)
ax.text(5.35, 0.22, "Development and validation", ha="center", color="#35526c", fontsize=8.5, weight="bold")
ax.text(12.08, 0.22, "Final held-out evaluation", ha="center", color="#88551a", fontsize=8.5, weight="bold")
ax.text(7.2, 2.82, "ClosureGraph-InSAR frozen experimental design", ha="center", fontsize=15, weight="bold")
ax.text(7.2, 2.45, "Checkpoints were selected from validation only; Cell 11 is read-only post-processing of saved outputs.",
        ha="center", fontsize=8.6, color="#404040")
produced["figures"] += save_figure(fig, "Fig_01_experimental_protocol")

# -------------------------------------------------------------------------
# SUPPLEMENTARY FIGURE S1 — Cell-8 provenance figures, assembled non-destructively.
# -------------------------------------------------------------------------
qc = optional_first("results/figures/*cell8*quality*.png", "results/figures/*quality*control*.png")
recon = optional_first("results/figures/*cell8*recon*.png", "results/figures/*compact*recon*.png")
inventory = []
for asset_name, source in (("dataset quality control", qc), ("compact reconstruction check", recon)):
    inventory.append({"asset": asset_name, "source": str(source) if source else "not found"})
if False and (qc or recon):
    count = int(qc is not None) + int(recon is not None)
    fig, axes = plt.subplots(1, count, figsize=(10 * count, 5.8))
    axes = np.atleast_1d(axes)
    for ax, (title, path) in zip(axes, (("Production dataset quality control", qc),
                                        ("Compact-shard reconstruction check", recon))):
        if path is None:
            continue
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title, weight="bold")
        ax.axis("off")
    fig.suptitle("Supplementary Figure S1 | Production-data validation", weight="bold", y=1.02)
    produced["figures"] += save_figure(fig, "Fig_S01_production_dataset_quality_control")
else:
    print("Cell-8 audit PNGs retained as source artifacts; omitted from the manuscript figure set.")

# -------------------------------------------------------------------------
# SUPPLEMENTARY FIGURE S2 — Cell-9R multi-seed validation history, if available.
# -------------------------------------------------------------------------
history_path = optional_first("results/tables/*cell9*repair*history*.csv", "results/tables/*cell9*history*.csv")
if False and history_path:
    history = pd.read_csv(history_path)
    epoch_c = column(history, "epoch")
    model_c = column(history, "model", "display_name", "phase")
    seed_c = column(history, "seed")
    score_c = column(history, "val_common_selection_score", "common_selection_score", "common_score")
    node_c = column(history, "val_node_mae", "node_mae")
    peak_c = column(history, "val_peak", "peak_loss", "peak")
    temp_c = column(history, "val_temporal", "temporal_gradient", "temporal")
    metrics = [(score_c, "Validation common score"), (node_c, "Validation node MAE"),
               (peak_c, "Validation peak-region loss"), (temp_c, "Validation temporal-gradient loss")]
    metrics = [(c, title) for c, title in metrics if c and epoch_c and model_c]
    if metrics:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        colors = {"Baseline": "#7f7f7f", "Graph + amplitude": "#009e73", "Graph Warm-Up (Internal)": "#377eb8"}
        for ax, (metric, title) in zip(axes.flat, metrics):
            for (model, seed), grp in history.groupby([model_c, seed_c] if seed_c else [model_c]):
                label_model = model_label(model if isinstance(model, str) else model[0])
                color = colors.get(label_model, "#4c78a8")
                label = f"{label_model} seed {seed}" if seed_c else label_model
                grp = grp.sort_values(epoch_c)
                ax.plot(grp[epoch_c], grp[metric], color=color, alpha=0.55, linewidth=1.4, label=label)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Normalized validation loss")
            ax.grid(alpha=0.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
        fig.suptitle("Supplementary Figure S2 | Multi-seed validation trajectories", weight="bold", fontsize=15)
        produced["figures"] += save_figure(fig, "Fig_S02_multiseed_training")
    else:
        print("Cell-9R history found but its columns were not recognised; Fig. 03 skipped.")
else:
    print("Cell-9R epoch histories retained as audit artifacts; omitted from the manuscript figure set.")

# -------------------------------------------------------------------------
# MAIN FIGURE 2 — focused locked-test effect-size plot.
# This replaces the six bar charts. It is easier to read and reports both
# direction and uncertainty without implying that every metric improved.
# -------------------------------------------------------------------------
model_c = column(summary, "model", "display_name")
if model_c is None:
    raise RuntimeError("Cell-10 summary has no model/display_name column.")
summary["_label"] = summary[model_c].map(model_label)
seed_model_c = column(seed_summary, "model", "display_name")
if seed_model_c:
    seed_summary["_label"] = seed_summary[seed_model_c].map(model_label)

metric_candidates = [
    ("mae_mm", "MAE", "mm"), ("rmse_mm", "RMSE", "mm"),
    ("ssim", "SSIM", ""), ("snr_linear", "SNR", "linear"),
    ("absolute_peak_bias_mm", "Absolute peak bias", "mm"),
    ("temporal_correlation", "Temporal correlation", ""),
    ("absolute_onset_error_epochs", "Onset error", "epochs"),
]
metrics = [(column(summary, name), title, unit) for name, title, unit in metric_candidates]
metrics = [(c, title) for c, title, _unit in metrics if c]
if not metrics:
    numeric = number_columns(summary)
    metrics = [(c, c.replace("_", " ").title()) for c in numeric[:6]]

labels = [x for x in ("Baseline", "Graph + amplitude") if x in set(summary["_label"])]
if len(labels) < 2:
    labels = list(summary["_label"].drop_duplicates())[:2]
palette = {"Baseline": "#5b5b5b", "Graph + amplitude": "#007f73"}
sig_metric_c = column(significance, "metric")
sig_low_c = column(significance, "bootstrap_ci_95_low")
sig_high_c = column(significance, "bootstrap_ci_95_high")
sig_holm_c = column(significance, "significant_holm_0_05")
effect_rows = []
for metric, title in metrics:
    if metric not in summary.columns:
        continue
    base = float(summary.loc[summary["_label"] == "Baseline", metric].iloc[0])
    graph = float(summary.loc[summary["_label"] == "Graph + amplitude", metric].iloc[0])
    lower = metric_direction(metric)
    favourable = (base - graph) if lower else (graph - base)
    low = high = np.nan
    significant = False
    if sig_metric_c:
        hit = significance.loc[significance[sig_metric_c].astype(str).str.lower() == metric.lower()]
        if not hit.empty:
            hit = hit.iloc[0]
            if sig_low_c and sig_high_c:
                # Cell 10 stores CIs in the same favourable orientation.
                low, high = float(hit[sig_low_c]), float(hit[sig_high_c])
            if sig_holm_c:
                significant = str(hit[sig_holm_c]).strip().lower() in {"true", "1", "yes"}
    scale = abs(base) if abs(base) > 1e-12 else 1.0
    effect_rows.append({"metric": metric, "label": title, "baseline": base, "graph": graph,
                        "favourable_change_pct": 100 * favourable / scale,
                        "ci_low_pct": 100 * low / scale, "ci_high_pct": 100 * high / scale,
                        "holm": significant, "direction": "lower" if lower else "higher"})
effect_frame = pd.DataFrame(effect_rows)
core_order = ["mae_mm", "rmse_mm", "ssim", "snr_linear", "absolute_peak_bias_mm", "temporal_correlation", "absolute_onset_error_epochs"]
effect_frame["order"] = effect_frame.metric.map({m: i for i, m in enumerate(core_order)})
effect_frame = effect_frame.sort_values("order", ascending=True).reset_index(drop=True)
fig = plt.figure(figsize=(12.6, 5.7), layout="constrained")
grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.65])
table_ax = fig.add_subplot(grid[0, 0])
ax = fig.add_subplot(grid[0, 1])
fig.suptitle("Locked-test performance of frozen three-seed ensembles", weight="bold", fontsize=15)

table_ax.axis("off")
table_ax.set_xlim(0, 1); table_ax.set_ylim(0, 1)
table_ax.text(0.00, 0.95, "Ensemble estimates", weight="bold", fontsize=10)
table_ax.text(0.60, 0.90, "Baseline", ha="right", fontsize=8, weight="bold")
table_ax.text(0.99, 0.90, "Graph + amp.", ha="right", fontsize=8, weight="bold")
for idx, row in effect_frame.iterrows():
    y = 0.82 - idx * 0.105
    table_ax.text(0.00, y, row["label"], fontsize=8.2, va="center")
    table_ax.text(0.60, y, f"{row['baseline']:.4g}", ha="right", fontsize=8.2, va="center")
    table_ax.text(0.99, y, f"{row['graph']:.4g}", ha="right", fontsize=8.2, va="center")
    table_ax.plot([0, 1], [y - 0.046, y - 0.046], color="#d7d7d7", lw=0.6)
table_ax.text(0.00, 0.045, "n = 2,000 locked-test sequences\nAll checkpoints frozen before test access.", fontsize=7.3, color="#444444")

for y, row in enumerate(effect_frame.itertuples(index=False)):
    colour = "#008f7a" if row.favourable_change_pct >= 0 else "#cc5a3c"
    if np.isfinite(row.ci_low_pct) and np.isfinite(row.ci_high_pct):
        ax.hlines(y, row.ci_low_pct, row.ci_high_pct, color=colour, lw=2.2, zorder=2)
        ax.vlines([row.ci_low_pct, row.ci_high_pct], y - 0.10, y + 0.10, color=colour, lw=1.2, zorder=2)
    ax.scatter(row.favourable_change_pct, y, s=58, color=colour, edgecolor="white", linewidth=0.9, zorder=3)
    significance_mark = "*" if row.holm else ""
    label_x = row.favourable_change_pct + (0.7 if row.favourable_change_pct >= 0 else -0.7)
    align = "left" if row.favourable_change_pct >= 0 else "right"
    ax.text(label_x, y, f"{row.favourable_change_pct:+.1f}%{significance_mark}", ha=align, va="center", fontsize=8)
ax.axvline(0, color="#202020", lw=1)
ax.set_yticks(range(len(effect_frame)), [f"{x.label} ({x.direction} better)" for x in effect_frame.itertuples(index=False)])
ax.invert_yaxis()
ax.set_xlabel("Change favourable to Graph + amplitude (%)")
ax.set_title("Effect estimates with 95% bootstrap confidence intervals", fontsize=10)
ax.grid(axis="x", alpha=0.20)
ax.text(0.01, -0.15, "* Holm-adjusted p < 0.05. Positive values favour Graph + amplitude.", transform=ax.transAxes, fontsize=7.5)
produced["figures"] += save_figure(fig, "Fig_02_locked_test_effects")
produced["tables"] += save_table(effect_frame.drop(columns="order"), "Table_02_main_locked_test_effect_sizes")

# -------------------------------------------------------------------------
# FIGURE 3 — variability across the three independently trained seeds.
# The Cell-10 records are deliberately NOT plotted here: the saved record
# identifier is not a unique sequence identifier, so per-sequence violins or
# scatter plots would be misleading.
# -------------------------------------------------------------------------
seed_id_c = column(seed_summary, "seed")
seed_metrics = [(column(seed_summary, "mae_mm", "mae"), "MAE (mm)", True),
                (column(seed_summary, "absolute_peak_bias_mm", "peak_bias"), "Absolute peak bias (mm)", True),
                (column(seed_summary, "absolute_onset_error_epochs", "onset_error"), "Onset error (epochs)", True),
                (column(seed_summary, "ssim"), "SSIM", False)]
seed_metrics = [(c, title, lower) for c, title, lower in seed_metrics if c]
if seed_id_c and seed_metrics and set(labels).issubset(set(seed_summary["_label"])):
    fig, axes = plt.subplots(1, len(seed_metrics), figsize=(3.25 * len(seed_metrics), 4.2), layout="constrained")
    axes = np.atleast_1d(axes)
    offsets = {"Baseline": -0.13, "Graph + amplitude": 0.13}
    for ax, (metric, title, lower) in zip(axes, seed_metrics):
        for label in labels:
            grp = seed_summary.loc[seed_summary["_label"] == label, [seed_id_c, metric]].dropna().sort_values(seed_id_c)
            x = np.full(len(grp), 1.0 + offsets[label])
            ax.scatter(x, grp[metric], s=52, color=palette.get(label, "#555555"), edgecolor="white", linewidth=0.8, zorder=3, label=label)
            ax.hlines(grp[metric].mean(), x.min() - 0.06, x.max() + 0.06, color=palette.get(label, "#555555"), lw=2.0, zorder=2)
        ax.set_xlim(0.60, 1.40)
        ax.set_xticks([1], ["Three independent seeds"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Lower is better" if lower else "Higher is better")
        ax.grid(axis="y", alpha=0.20)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Independent-seed reproducibility on the locked test", weight="bold", fontsize=14)
    produced["figures"] += save_figure(fig, "Fig_03_independent_seed_results")
else:
    print("Seed-level identifiers/metrics not recognised; Fig. 03 skipped.")

# -------------------------------------------------------------------------
# TABLES AND MANUSCRIPT CAPTIONS
# -------------------------------------------------------------------------
protocol = pd.DataFrame([
    ["Production dataset", "20,000 train; 2,000 validation; 2,000 locked test", "Cell 8"],
    ["Independent training", "3 random seeds for baseline and Graph + amplitude", "Cell 9R"],
    ["Checkpoint selection", "Frozen validation common score", "Cell 9R"],
    ["Final evaluation", "Locked test opened exactly once; paired analysis", "Cell 10"],
    ["Manuscript assets", "Read-only rendering of saved outputs", "Cell 11"],
], columns=["Component", "Protocol", "Source"])
produced["tables"] += save_table(protocol, "Table_01_frozen_experimental_protocol")
publication_results = effect_frame.loc[:, ["label", "baseline", "graph", "favourable_change_pct", "ci_low_pct", "ci_high_pct", "holm", "direction"]].copy()
publication_results.columns = ["Metric", "Baseline", "Graph + amplitude", "Favourable change (%)", "95% CI low (%)", "95% CI high (%)", "Holm significant", "Favourable direction"]
produced["tables"] += save_table(publication_results, "Table_02_main_locked_test_results")
produced["tables"] += save_table(summary.drop(columns=["_label"], errors="ignore"), "Table_S02_full_locked_test_summary")
produced["tables"] += save_table(significance, "Table_S03_paired_locked_test_significance")
produced["tables"] += save_table(seed_summary.drop(columns=["_label"], errors="ignore"), "Table_S04_locked_test_seed_summary")
asset_inventory = pd.DataFrame(inventory)
produced["tables"] += save_table(asset_inventory, "Table_S01_asset_inventory")

captions = """# ClosureGraph-InSAR manuscript captions (version 3)

## Figure 1. Study design and leakage-control protocol.
The production experiment uses 20,000 training sequences, 2,000 validation sequences, and 2,000 locked-test sequences. Baseline and Graph + amplitude models are trained with three independent seeds. Checkpoints are selected using the validation common score only. The locked test is opened exactly once after checkpoints are frozen. Cell 11 is read-only post-processing of saved artifacts.

## Figure 2. Locked-test effect estimates.
Favourable percentage changes of Graph + amplitude relative to the baseline on the 2,000-sequence locked test. For lower-is-better measures the difference is baseline minus Graph + amplitude; for higher-is-better measures it is Graph + amplitude minus baseline. Horizontal intervals are 95% bootstrap confidence intervals calculated in Cell 10. Asterisks indicate Holm-adjusted p < 0.05. Results are descriptive of the frozen, one-time locked-test evaluation.

## Figure 3. Independent-seed reproducibility.
Locked-test scores for the three independently trained seeds, with horizontal lines showing the seed mean. This figure quantifies reproducibility of the frozen models without incorrectly treating non-unique saved record identifiers as individual sequences.

## Table 1. Frozen experimental protocol.
The data split, independent-seed training, validation-only checkpoint selection, and one-time locked-test evaluation protocol.

## Table 2. Main locked-test results.
Baseline and Graph + amplitude ensemble performance, favourable relative change, bootstrap confidence intervals, and Holm-corrected significance. The table should be used as the primary quantitative results table.

## Supplementary Table S2. Full locked-test summary.
Unabridged aggregate metrics saved by Cell 10.

## Supplementary Table S3. Paired locked-test significance.
Paired inferential results generated by Cell 10 using the pre-specified testing protocol.

## Supplementary Table S4. Seed-level locked-test performance.
Results for the independently trained model seeds used to quantify training variability.
"""
caption_path = OUT / "closuregraph_manuscript_captions_v3.md"
caption_path.write_text(captions, encoding="utf-8")

manifest = {
    "version": VERSION,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": "read-only post-processing; no training and no locked-test shard access",
    "figure_scope": "protocol, aggregate locked-test effects, and seed-level reproducibility only",
    "known_limit": "No qualitative prediction maps are generated because the saved Cell-10 CSVs do not contain image predictions.",
    "cell10_complete_audit": str(complete_path),
    "inputs": {"summary": str(summary_path), "seed_summary": str(seed_path),
               "significance": str(sig_path), "records": str(records_path),
               "history": str(history_path) if history_path else None},
    "outputs": produced,
    "captions": str(caption_path),
}
manifest_path = OUT_LOG / "closuregraph_cell11_manuscript_assets_v3.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

banner("CELL 11 COMPLETE: MANUSCRIPT ASSET PACKAGE")
print(f"Figures  : {OUT_FIG}")
print(f"Tables   : {OUT_TAB}")
print(f"Captions : {caption_path}")
print(f"Manifest : {manifest_path}")
print("Use the PDF/SVG files for manuscript layout and PNG files for Word/PowerPoint drafts.")
