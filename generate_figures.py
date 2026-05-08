#!/usr/bin/env python3
"""
Generate all experiment figures for the thesis.
Usage: python generate_figures.py
Output: figures/ directory with PDF/PNG files.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------------
# Style setup (from plot-style.md, adapted for non-LaTeX env)
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

HAS_LATEX = os.system("which pdflatex > /dev/null 2>&1") == 0
if HAS_LATEX:
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
else:
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "cm"

FONT_SIZE = 16
plt.rc("axes", titlesize=FONT_SIZE, labelsize=FONT_SIZE)
plt.rc("xtick", labelsize=FONT_SIZE - 2)
plt.rc("ytick", labelsize=FONT_SIZE - 2)
plt.rc("legend", fontsize=FONT_SIZE - 2)
plt.rc("figure", titlesize=FONT_SIZE + 2)

COLORS = {
    "exp_a": "#7f7f7f",     # gray
    "exp_b": "#9467bd",     # purple
    "exp_c": "#1f77b4",     # blue (ours)
    "exp_d": "#ff7f0e",     # orange (oracle)
    "successful": "#2ca02c",  # green
    "failure": "#d62728",     # red
}

ROBOMETER_ROOT = Path("/root/autodl-tmp/robometer")
EVAL_OUT = ROBOMETER_ROOT / "baseline_eval_output"
EVAL_OUT_2B = ROBOMETER_ROOT / "baseline_eval_output-2b"
LOGS_ROOT = ROBOMETER_ROOT / "logs"
PROCESSED_DS = Path("/root/autodl-tmp/processed_datasets")
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_all_metrics(exp_dir):
    """Load all_metrics.json if it exists, otherwise build from individual metrics files."""
    all_metrics_path = Path(exp_dir) / "all_metrics.json"
    if all_metrics_path.exists():
        return load_json(all_metrics_path)
    result = {}
    ra_metrics = Path(exp_dir) / "reward_alignment" / "metrics.json"
    if ra_metrics.exists():
        result["reward_alignment"] = load_json(ra_metrics)
    pr_metrics = Path(exp_dir) / "policy_ranking" / "metrics.json"
    if pr_metrics.exists():
        result["policy_ranking"] = load_json(pr_metrics)
    return result


def load_npz_frames(npz_path):
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    key = list(data.keys())[0]
    frames = data[key]
    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        return frames
    if frames.ndim == 4 and frames.shape[1] in (1, 3):
        return frames.transpose(0, 2, 3, 1)
    return frames


# ===================================================================
# Figure 1: LIBERO Environment Overview
# ===================================================================
def gen_fig1_libero_overview():
    """Grid of representative LIBERO task frames."""
    results_file = EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000" / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    data = load_json(results_file)

    successful = [d for d in data if d["quality_label"] == "successful"]
    successful.sort(key=lambda x: x["task"])

    seen_tasks = set()
    unique_task_trajs = []
    for d in successful:
        if d["task"] not in seen_tasks:
            seen_tasks.add(d["task"])
            unique_task_trajs.append(d)

    n_tasks = min(8, len(unique_task_trajs))
    selected = unique_task_trajs[:n_tasks]

    cols = 4
    rows = (n_tasks + cols - 1) // cols

    all_variants = []
    for variant_idx in range(3):
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

        loaded_count = 0
        for i, traj in enumerate(selected):
            video_path = traj["video_path"]
            frames = load_npz_frames(video_path)
            if frames is None:
                axes[i].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
                axes[i].set_title(traj["task"][:40] + "...", fontsize=10, pad=4)
                axes[i].axis("off")
                continue

            pick_indices = [0, len(frames) // 3, 2 * len(frames) // 3]
            frame_idx = pick_indices[min(variant_idx, len(pick_indices) - 1)]
            frame_idx = min(frame_idx, len(frames) - 1)
            frame = frames[frame_idx]

            axes[i].imshow(frame)
            task_short = traj["task"]
            if len(task_short) > 45:
                task_short = task_short[:42] + "..."
            axes[i].set_title(task_short, fontsize=9, pad=4)
            axes[i].axis("off")
            loaded_count += 1

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle("LIBERO-90 Benchmark Tasks (Unseen Evaluation Set)", fontsize=FONT_SIZE, y=1.02)
        plt.tight_layout()

        out_path = OUT_DIR / f"fig1_libero_overview_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig1 v{variant_idx+1}] Saved {out_path} ({loaded_count} tasks loaded)")

    return all_variants


# ===================================================================
# Figure 2: Progress Curves — Successful vs Failed (Ours)
# ===================================================================
def gen_fig2_progress_curves():
    """Progress curves for successful vs failed trajectories, same task."""
    results_90 = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000" / "policy_ranking" / "libero_90_libero_90_failure_task_groups.json"
    )

    all_variants = []
    tasks_with_both = []
    for task, items in results_90.items():
        labels = [it["quality_label"] for it in items]
        if "successful" in labels and "failure" in labels:
            tasks_with_both.append(task)

    tasks_with_both.sort()
    n_show = min(6, len(tasks_with_both))
    show_tasks = tasks_with_both[:n_show]

    for variant_idx in range(2):
        cols = min(3, n_show)
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, task in enumerate(show_tasks):
            ax = axes[i]
            items = results_90[task]

            succ_items = [it for it in items if it["quality_label"] == "successful"]
            fail_items = [it for it in items if it["quality_label"] == "failure"]

            agg_key = "final_predicted_reward_avg" if variant_idx == 0 else "final_predicted_reward_sum"

            for si, s in enumerate(succ_items[:2]):
                val = s[agg_key]
                ax.bar(f"S{si+1}", val, color=COLORS["successful"], alpha=0.8, edgecolor="black", linewidth=0.5)
            for fi, f_item in enumerate(fail_items[:2]):
                val = f_item[agg_key]
                ax.bar(f"F{fi+1}", val, color=COLORS["failure"], alpha=0.8, edgecolor="black", linewidth=0.5)

            task_short = task if len(task) <= 35 else task[:32] + "..."
            ax.set_title(task_short, fontsize=9, pad=4)
            ax.set_ylabel("Predicted Reward" if i % cols == 0 else "")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        agg_label = "Average" if variant_idx == 0 else "Sum"
        fig.suptitle(f"Ours (BT+MaxEnt): Predicted Rewards — Successful vs Failed ({agg_label} Aggregation)", fontsize=FONT_SIZE)
        plt.tight_layout()
        out_path = OUT_DIR / f"fig2_reward_succ_vs_fail_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig2 v{variant_idx+1}] Saved {out_path}")

    # Variant 3: per-frame progress curves from reward_alignment results
    results_ra = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000" / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )
    # Group by task
    task_to_trajs = defaultdict(list)
    for d in results_ra:
        task_to_trajs[d["task"]].append(d)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes = axes.flatten()
    for i, (task, trajs) in enumerate(sorted(task_to_trajs.items())[:10]):
        ax = axes[i]
        for traj in trajs:
            pp = traj["progress_pred"]
            color = COLORS["successful"] if traj["quality_label"] == "successful" else COLORS["failure"]
            label = traj["quality_label"]
            ax.plot(range(len(pp)), pp, color=color, linewidth=2, alpha=0.8, label=label)
        task_short = task if len(task) <= 30 else task[:27] + "..."
        ax.set_title(task_short, fontsize=8, pad=3)
        ax.set_ylim(-0.1, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i >= 5:
            ax.set_xlabel("Frame", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("Progress", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color=COLORS["successful"], lw=2, label="Successful"),
        plt.Line2D([0], [0], color=COLORS["failure"], lw=2, label="Failed"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Ours (BT+MaxEnt): Per-Frame Progress Predictions on LIBERO-90", fontsize=FONT_SIZE, y=1.06)
    plt.tight_layout()
    out_path = OUT_DIR / "fig2_progress_curves_v3.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig2 v3] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 3: Ours vs Supervised Oracle — Side-by-side progress curves
# ===================================================================
def gen_fig3_exp_c_vs_d():
    """Side-by-side progress curves for same trajectories under Exp C vs D."""
    results_c = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000" / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )
    results_d = load_json(
        EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900" / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )

    c_by_id = {d["id"]: d for d in results_c}
    d_by_id = {d["id"]: d for d in results_d}
    common_ids = sorted(set(c_by_id.keys()) & set(d_by_id.keys()))

    all_variants = []
    n_show = min(10, len(common_ids))

    # Variant 1: all trajectories in a grid
    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, tid in enumerate(common_ids[:n_show]):
        ax = axes[i]
        c_traj = c_by_id[tid]
        d_traj = d_by_id[tid]

        pp_c = c_traj["progress_pred"]
        pp_d = d_traj["progress_pred"]

        ax.plot(range(len(pp_c)), pp_c, color=COLORS["exp_c"], linewidth=2.5, label="Ours (BT+MaxEnt)")
        ax.plot(range(len(pp_d)), pp_d, color=COLORS["exp_d"], linewidth=2.5, label="Supervised Oracle", linestyle="--")

        # Ground truth if available
        if "target_progress" in c_traj and c_traj["target_progress"]:
            tp = c_traj["target_progress"]
            ax.plot(range(len(tp)), tp, color="gray", linewidth=1, linestyle=":", alpha=0.5, label="GT")

        task_short = c_traj["task"]
        if len(task_short) > 30:
            task_short = task_short[:27] + "..."
        ql = c_traj["quality_label"]
        ax.set_title(f"{task_short}\n({ql})", fontsize=8, pad=3)
        ax.set_ylim(-0.1, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i >= cols:
            ax.set_xlabel("Frame", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("Progress", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=COLORS["exp_c"], lw=2.5, label="Ours (BT+MaxEnt)"),
        plt.Line2D([0], [0], color=COLORS["exp_d"], lw=2.5, linestyle="--", label="Supervised Oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Progress Predictions: Ours vs Supervised Oracle on Same Trajectories", fontsize=FONT_SIZE, y=1.06)
    plt.tight_layout()
    out_path = OUT_DIR / "fig3_exp_c_vs_d_v1.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig3 v1] Saved {out_path}")

    # Variant 2: 2x3 selected subset, larger
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, tid in enumerate(common_ids[:6]):
        ax = axes[i]
        c_traj = c_by_id[tid]
        d_traj = d_by_id[tid]

        pp_c = c_traj["progress_pred"]
        pp_d = d_traj["progress_pred"]

        ax.plot(range(len(pp_c)), pp_c, color=COLORS["exp_c"], linewidth=2.5, label="Ours (BT+MaxEnt)")
        ax.plot(range(len(pp_d)), pp_d, color=COLORS["exp_d"], linewidth=2.5, linestyle="--", label="Supervised Oracle")

        if "target_progress" in c_traj and c_traj["target_progress"]:
            tp = c_traj["target_progress"]
            ax.plot(range(len(tp)), tp, color="gray", linewidth=1, linestyle=":", alpha=0.5, label="GT")

        task_short = c_traj["task"]
        if len(task_short) > 40:
            task_short = task_short[:37] + "..."
        ax.set_title(f"{task_short} ({c_traj['quality_label']})", fontsize=11, pad=5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Progress" if i % 3 == 0 else "", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=COLORS["exp_c"], lw=2.5, label="Ours (BT+MaxEnt)"),
        plt.Line2D([0], [0], color=COLORS["exp_d"], lw=2.5, linestyle="--", label="Supervised Oracle"),
        plt.Line2D([0], [0], color="gray", lw=1, linestyle=":", alpha=0.5, label="Ground Truth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    out_path = OUT_DIR / "fig3_exp_c_vs_d_v2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig3 v2] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 4: Main Results Bar Chart
# ===================================================================
def gen_fig4_bar_chart():
    """Grouped bar chart of main quantitative results across experiments."""
    metrics_c = load_all_metrics(EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000")
    metrics_d = load_all_metrics(EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900")

    all_variants = []

    # ---- Variant 1: LIBERO-90 metrics ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    metric_configs = [
        ("VOC r (Reward Alignment)", "reward_alignment", "libero_90_libero_90_failure/pearson", False),
        ("Kendall tau (Policy Ranking)", "policy_ranking", "libero_90_libero_90_failure/kendall_sum", False),
        ("Ranking Accuracy", "policy_ranking", "libero_90_libero_90_failure/ranking_acc_sum", False),
        ("Succ-Fail Diff", "policy_ranking", "libero_90_libero_90_failure/avg_succ_fail_diff_sum", False),
    ]

    methods = ["Ours\n(BT+MaxEnt)", "Supervised\nOracle"]
    method_colors = [COLORS["exp_c"], COLORS["exp_d"]]
    x = np.arange(len(methods))
    width = 0.5

    for ax_idx, (title, section, key, _) in enumerate(metric_configs):
        ax = axes[ax_idx]
        vals = []
        for m_data in [metrics_c, metrics_d]:
            if section in m_data and key in m_data[section]:
                vals.append(m_data[section][key])
            else:
                vals.append(0)

        bars = ax.bar(x, vals, width, color=method_colors, edgecolor="black", linewidth=0.8)

        for bar, val in zip(bars, vals):
            y_off = 0.02 * max(abs(v) for v in vals) if vals else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("LIBERO-90: Ours (BT+MaxEnt) vs Supervised Oracle", fontsize=FONT_SIZE, y=1.03)
    plt.tight_layout()
    out_path = OUT_DIR / "fig4_main_results_bar_v1.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig4 v1] Saved {out_path}")

    # ---- Variant 2: LIBERO-10 + LIBERO-90 combined ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (dataset_label, dataset_key) in enumerate([
        ("LIBERO-90", "libero_90_libero_90_failure"),
        ("LIBERO-10", "libero_10_libero_10_failure"),
    ]):
        ax = axes[ax_idx]
        metric_names = ["VOC r", "Kendall tau", "Ranking Acc"]
        metric_keys = [
            f"{dataset_key}/pearson",
            f"{dataset_key}/kendall_sum",
            f"{dataset_key}/ranking_acc_sum",
        ]
        sections = ["reward_alignment", "policy_ranking", "policy_ranking"]

        x = np.arange(len(metric_names))
        w = 0.3
        for mi, (m_data, m_label, m_color) in enumerate([
            (metrics_c, "Ours (BT+MaxEnt)", COLORS["exp_c"]),
            (metrics_d, "Supervised Oracle", COLORS["exp_d"]),
        ]):
            vals = []
            for sec, key in zip(sections, metric_keys):
                if sec in m_data and key in m_data[sec]:
                    vals.append(m_data[sec][key])
                else:
                    vals.append(0)
            offset = (mi - 0.5) * w
            bars = ax.bar(x + offset, vals, w * 0.9, label=m_label, color=m_color,
                         edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(dataset_label, fontsize=14, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(-0.2, 1.15)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=11)

    fig.suptitle("Reward Model Evaluation: Ours vs Supervised Oracle", fontsize=FONT_SIZE, y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / "fig4_main_results_bar_v2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig4 v2] Saved {out_path}")

    # ---- Variant 3: All 4 methods (A/B/C/D) — SmolVLM runs ----
    metrics_a_dir = EVAL_OUT / "rbm_exp_a_pure_bt_smolvlm_checkpoint-1000"
    metrics_b_dir = EVAL_OUT / "rbm_exp_b_l2_smooth_smolvlm_checkpoint-1000"

    metrics_a = load_all_metrics(metrics_a_dir)
    metrics_b = load_all_metrics(metrics_b_dir)

    if metrics_a and metrics_b:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        metric_configs_4 = [
            ("VOC r", "reward_alignment", "libero_90_libero_90_failure/pearson"),
            ("Kendall tau", "policy_ranking", "libero_90_libero_90_failure/kendall_sum"),
            ("Ranking Accuracy", "policy_ranking", "libero_90_libero_90_failure/ranking_acc_sum"),
            ("Succ-Fail Diff", "policy_ranking", "libero_90_libero_90_failure/avg_succ_fail_diff_sum"),
        ]
        all_methods = [
            ("Pure BT", metrics_a, COLORS["exp_a"]),
            ("BT+L2\nSmooth", metrics_b, COLORS["exp_b"]),
            ("Ours\n(BT+MaxEnt)", metrics_c, COLORS["exp_c"]),
            ("Supervised\nOracle", metrics_d, COLORS["exp_d"]),
        ]

        for ax_idx, (title, section, key) in enumerate(metric_configs_4):
            ax = axes[ax_idx]
            x = np.arange(len(all_methods))
            vals = []
            colors = []
            labels = []
            for label, m_data, color in all_methods:
                if section in m_data and key in m_data[section]:
                    vals.append(m_data[section][key])
                else:
                    vals.append(0)
                colors.append(color)
                labels.append(label)

            bars = ax.bar(x, vals, 0.55, color=colors, edgecolor="black", linewidth=0.8)
            for bar, val in zip(bars, vals):
                y_off = 0.02 * max(abs(v) for v in vals) if vals else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=13, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("LIBERO-90: Ablation Study \u2014 Four Methods Compared", fontsize=FONT_SIZE, y=1.03)
        plt.tight_layout()
        out_path = OUT_DIR / "fig4_main_results_bar_v3.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig4 v3] Saved {out_path}")
    else:
        print("  [Fig4 v3] Skipped — Exp A/B eval results not yet available")

    return all_variants


# ===================================================================
# Figure 5: Training Dynamics from TensorBoard
# ===================================================================
def gen_fig5_training_dynamics():
    """Training loss curves from TensorBoard data (SmolVLM only)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    all_variants = []

    exp_configs = {
        "Ours (BT+MaxEnt)": {
            "tb_dir": str(LOGS_ROOT / "exp_c_dirfix_v1" / "exp_c_dirfix_v1" / "tb"),
            "color": COLORS["exp_c"],
            "tags": {
                "Preference Loss": "train/preference_loss",
                "Struct Loss": "train/struct_loss",
                "Preference Accuracy": "train/preference_accuracy",
                "Delta Variance": "train/delta_variance",
                "Delta Mean": "train/delta_mean",
            }
        },
        "Supervised Oracle (RoboMeter)": {
            "tb_dir": str(LOGS_ROOT / "exp_d_robometer_smolvlm" / "exp_d_robometer_smolvlm" / "tb"),
            "color": COLORS["exp_d"],
            "tags": {
                "Preference Loss": "train/preference_loss",
                "Preference Accuracy": "train/preference_accuracy",
                "Progress Loss": "train/pref_prog_loss",
                "Progress Spearman": "train/pref_prog_spearman_corr",
            }
        },
    }

    # Load all data
    all_data = {}
    for exp_name, config in exp_configs.items():
        ea = EventAccumulator(config["tb_dir"])
        ea.Reload()
        exp_data = {}
        for tag_label, tag_name in config["tags"].items():
            try:
                events = ea.Scalars(tag_name)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                exp_data[tag_label] = (steps, values)
            except KeyError:
                pass
        all_data[exp_name] = exp_data

    # ---- Variant 1: Exp C training dynamics (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    c_data = all_data["Ours (BT+MaxEnt)"]
    c_tags = ["Preference Loss", "Struct Loss", "Preference Accuracy", "Delta Variance"]
    c_ylabels = ["Loss", "Entropy Loss", "Accuracy", "Variance"]

    for i, (tag, ylabel) in enumerate(zip(c_tags, c_ylabels)):
        ax = axes[i]
        if tag in c_data:
            steps, values = c_data[tag]
            ax.plot(steps, values, color=COLORS["exp_c"], alpha=0.3, linewidth=0.8)
            smoothed = smooth(values, weight=0.8)
            ax.plot(steps, smoothed, color=COLORS["exp_c"], linewidth=2.5, label="Ours")
        ax.set_title(tag, fontsize=13, pad=6)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if "Struct Loss" in c_data:
        axes[1].axhline(y=-np.log(7), color="red", linestyle="--", linewidth=1, alpha=0.5, label="Saturation: -log(7)")
        axes[1].legend(fontsize=10)

    fig.suptitle("Ours (BT+MaxEnt) Training Dynamics (SmolVLM-500M)", fontsize=FONT_SIZE, y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v1.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v1] Saved {out_path}")

    # ---- Variant 2: Exp D training dynamics (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    d_data = all_data["Supervised Oracle (RoboMeter)"]
    d_tags = ["Preference Loss", "Preference Accuracy", "Progress Loss", "Progress Spearman"]
    d_ylabels = ["Loss", "Accuracy", "Loss", "Spearman Corr"]

    for i, (tag, ylabel) in enumerate(zip(d_tags, d_ylabels)):
        ax = axes[i]
        if tag in d_data:
            steps, values = d_data[tag]
            ax.plot(steps, values, color=COLORS["exp_d"], alpha=0.3, linewidth=0.8)
            smoothed = smooth(values, weight=0.8)
            ax.plot(steps, smoothed, color=COLORS["exp_d"], linewidth=2.5)
        ax.set_title(tag, fontsize=13, pad=6)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Supervised Oracle Training Dynamics (SmolVLM-500M)", fontsize=FONT_SIZE, y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v2] Saved {out_path}")

    # ---- Variant 3: Combined — Exp C vs D overlaid preference loss + accuracy ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, tag in enumerate(["Preference Loss", "Preference Accuracy"]):
        ax = axes[ax_idx]
        for exp_name, exp_data in all_data.items():
            if tag in exp_data:
                steps, values = exp_data[tag]
                color = exp_configs[exp_name]["color"]
                ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.6)
                smoothed = smooth(values, weight=0.8)
                ax.plot(steps, smoothed, color=color, linewidth=2.5, label=exp_name)
        ax.set_title(tag, fontsize=14, pad=6)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Training Comparison: Ours vs Supervised Oracle (SmolVLM-500M)", fontsize=FONT_SIZE, y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v3.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v3] Saved {out_path}")

    # ---- Variant 4: Key Exp C figure for thesis — struct_loss + BT loss dual y-axis ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    if "Preference Loss" in c_data:
        steps, values = c_data["Preference Loss"]
        smoothed = smooth(values, weight=0.85)
        ax1.plot(steps, smoothed, color=COLORS["exp_c"], linewidth=2.5, label=r"$\mathcal{L}_{BT}$")
        ax1.plot(steps, values, color=COLORS["exp_c"], alpha=0.15, linewidth=0.6)

    if "Struct Loss" in c_data:
        steps, values = c_data["Struct Loss"]
        smoothed = smooth(values, weight=0.85)
        ax2.plot(steps, smoothed, color="#e377c2", linewidth=2.5, label=r"$\mathcal{L}_{struct}$")
        ax2.plot(steps, values, color="#e377c2", alpha=0.15, linewidth=0.6)
        ax2.axhline(y=-np.log(7), color="red", linestyle="--", linewidth=1, alpha=0.4)
        ax2.text(steps[-1] * 0.7, -np.log(7) + 0.03, "Saturation: -ln(7)", fontsize=9, color="red", alpha=0.7)

    ax1.set_xlabel("Training Step", fontsize=14)
    ax1.set_ylabel(r"$\mathcal{L}_{BT}$ (Preference Loss)", fontsize=14, color=COLORS["exp_c"])
    ax2.set_ylabel(r"$\mathcal{L}_{struct}$ (Entropy Loss)", fontsize=14, color="#e377c2")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper right")

    fig.suptitle("Ours (BT+MaxEnt): Joint Loss Training Dynamics", fontsize=FONT_SIZE, y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v4_dual_axis.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v4] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 6: Confusion Matrix Heatmap
# ===================================================================
def gen_fig6_confusion_matrix():
    """Task-instruction confusion matrix from policy ranking data."""
    all_variants = []

    for variant_idx, (exp_label, exp_dir) in enumerate([
        ("Ours (BT+MaxEnt)", EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"),
        ("Supervised Oracle", EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900"),
    ]):
        task_groups_file = exp_dir / "policy_ranking" / "libero_90_libero_90_failure_task_groups.json"
        if not task_groups_file.exists():
            continue

        task_groups = load_json(task_groups_file)
        tasks = sorted(task_groups.keys())
        n_tasks = len(tasks)

        task_to_idx = {t: i for i, t in enumerate(tasks)}
        matrix = np.zeros((n_tasks, n_tasks))
        count_matrix = np.zeros((n_tasks, n_tasks))

        for task, items in task_groups.items():
            tidx = task_to_idx[task]
            for item in items:
                reward = item["final_predicted_reward_avg"]
                matrix[tidx, tidx] += reward
                count_matrix[tidx, tidx] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.divide(matrix, count_matrix, out=np.zeros_like(matrix), where=count_matrix != 0)

        fig, ax = plt.subplots(figsize=(6, 5))
        task_labels_short = [t[:25] + "..." if len(t) > 25 else t for t in tasks]

        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                    xticklabels=task_labels_short, yticklabels=task_labels_short,
                    cbar_kws={"label": "Avg Predicted Reward"})
        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel("Task", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        fig.suptitle(f"Reward Heatmap: {exp_label}", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = OUT_DIR / f"fig6_confusion_matrix_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig6 v{variant_idx+1}] Saved {out_path}")

    # Variant 3: Side-by-side Exp C vs D
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (exp_label, exp_dir) in enumerate([
        ("Ours (BT+MaxEnt)", EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"),
        ("Supervised Oracle", EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900"),
    ]):
        ax = axes[ax_idx]
        task_groups_file = exp_dir / "policy_ranking" / "libero_90_libero_90_failure_task_groups.json"
        if not task_groups_file.exists():
            continue

        task_groups = load_json(task_groups_file)
        tasks = sorted(task_groups.keys())

        succ_rewards = []
        fail_rewards = []
        task_labels = []
        for task in tasks:
            items = task_groups[task]
            s_vals = [it["final_predicted_reward_avg"] for it in items if it["quality_label"] == "successful"]
            f_vals = [it["final_predicted_reward_avg"] for it in items if it["quality_label"] == "failure"]
            if s_vals and f_vals:
                succ_rewards.append(np.mean(s_vals))
                fail_rewards.append(np.mean(f_vals))
                task_labels.append(task[:20] + "..." if len(task) > 20 else task)

        x = np.arange(len(task_labels))
        w = 0.35
        ax.bar(x - w / 2, succ_rewards, w, label="Successful", color=COLORS["successful"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, fail_rewards, w, label="Failed", color=COLORS["failure"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(exp_label, fontsize=13)
        ax.set_ylabel("Avg Predicted Reward", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Reward Separation: Successful vs Failed per Task", fontsize=FONT_SIZE, y=1.03)
    plt.tight_layout()
    out_path = OUT_DIR / "fig6_reward_separation_v3.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig6 v3] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 7: Trajectory Key Frames with Progress Curves
# ===================================================================
def gen_fig7_trajectory_frames():
    """Key frames from trajectories with corresponding progress curves."""
    results_c = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000" / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )

    seen_tasks = set()
    unique_task_trajs = []
    for d in results_c:
        if d["task"] not in seen_tasks:
            seen_tasks.add(d["task"])
            unique_task_trajs.append(d)

    all_variants = []
    n_trajs = min(4, len(unique_task_trajs))

    for variant_idx in range(2):
        start_idx = variant_idx * n_trajs
        selected = unique_task_trajs[start_idx:start_idx + n_trajs]
        if not selected:
            break

        fig = plt.figure(figsize=(16, 4 * n_trajs))
        gs = GridSpec(n_trajs, 5, figure=fig, width_ratios=[1, 1, 1, 1, 2])

        for row, traj in enumerate(selected):
            video_path = traj["video_path"]
            frames = load_npz_frames(video_path)
            progress = traj["progress_pred"]
            task = traj["task"]
            ql = traj["quality_label"]

            if frames is not None:
                n_frames = len(frames)
                pick = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
                for col, fi in enumerate(pick):
                    ax = fig.add_subplot(gs[row, col])
                    ax.imshow(frames[fi])
                    ax.set_title(f"Frame {fi}", fontsize=9)
                    ax.axis("off")
            else:
                for col in range(4):
                    ax = fig.add_subplot(gs[row, col])
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
                    ax.axis("off")

            ax_plot = fig.add_subplot(gs[row, 4])
            color = COLORS["successful"] if ql == "successful" else COLORS["failure"]
            ax_plot.plot(range(len(progress)), progress, color=color, linewidth=2.5)
            ax_plot.set_ylim(-0.1, 1.1)
            ax_plot.set_ylabel("Progress", fontsize=11)
            ax_plot.set_xlabel("Frame" if row == n_trajs - 1 else "", fontsize=11)
            task_short = task if len(task) <= 40 else task[:37] + "..."
            ax_plot.set_title(f"{task_short} ({ql})", fontsize=10, pad=4)
            ax_plot.spines["top"].set_visible(False)
            ax_plot.spines["right"].set_visible(False)

            if "target_progress" in traj and traj["target_progress"]:
                tp = traj["target_progress"]
                ax_plot.plot(range(len(tp)), tp, color="gray", linewidth=1, linestyle=":", alpha=0.5)

        fig.suptitle("Trajectory Frames and Progress Predictions (Ours)", fontsize=FONT_SIZE, y=1.01)
        plt.tight_layout()
        out_path = OUT_DIR / f"fig7_trajectory_frames_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig7 v{variant_idx+1}] Saved {out_path}")

    return all_variants


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Generating thesis experiment figures")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    results = {}

    print("\n--- Figure 1: LIBERO Environment Overview ---")
    results["fig1"] = gen_fig1_libero_overview()

    print("\n--- Figure 2: Progress Curves (Successful vs Failed) ---")
    results["fig2"] = gen_fig2_progress_curves()

    print("\n--- Figure 3: Ours vs Supervised Oracle Comparison ---")
    results["fig3"] = gen_fig3_exp_c_vs_d()

    print("\n--- Figure 4: Main Results Bar Chart ---")
    results["fig4"] = gen_fig4_bar_chart()

    print("\n--- Figure 5: Training Dynamics ---")
    results["fig5"] = gen_fig5_training_dynamics()

    print("\n--- Figure 6: Confusion Matrix / Reward Heatmap ---")
    results["fig6"] = gen_fig6_confusion_matrix()

    print("\n--- Figure 7: Trajectory Frames with Progress ---")
    results["fig7"] = gen_fig7_trajectory_frames()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    total = sum(len(v) for v in results.values())
    print(f"Total files: {total}")
    for fig_name, paths in results.items():
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
