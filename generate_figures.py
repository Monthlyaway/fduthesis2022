#!/usr/bin/env python3
"""
Generate all experiment figures for the thesis.
Usage: python generate_figures.py
Output: figures/ directory with PDF/PNG files.

Color palette: Top-conference pastel / Morandi tones.
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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

# ===================================================================
# Dual-Layer Pastel Color System (Morandi / Top-Conference Style)
# ===================================================================
# Each hue has a "stroke" (deeper, for lines/edges) and "fill" (lighter,
# for bar fills / backgrounds).  This keeps lines readable on white while
# bar fills stay soft.

C = {
    # --- Method colors (4 experiments) ---
    "exp_a_stroke": "#9E9E9E",   "exp_a_fill": "#D5D5D5",   # Pure BT – neutral gray
    "exp_b_stroke": "#A08CB5",   "exp_b_fill": "#CAC2D7",   # BT+L2 – purple
    "exp_c_stroke": "#6DAFC2",   "exp_c_fill": "#BFDCE7",   # Ours (BT+MaxEnt) – blue
    "exp_d_stroke": "#E0A07A",   "exp_d_fill": "#F5D8BF",   # Supervised Oracle – peach

    # --- Trajectory quality ---
    "succ_stroke":  "#7EAE6A",   "succ_fill":  "#D9E4C2",   # Successful – sage green
    "fail_stroke":  "#D47B7B",   "fail_fill":  "#F0C4C4",   # Failed – soft coral

    # --- Accent ---
    "struct_stroke": "#C48DBF",  "struct_fill": "#E0C4DC",   # Structural loss – muted magenta
    "gt":            "#AAAAAA",                               # Ground truth – light gray

    # --- Text & chrome ---
    "text":   "#333333",
    "spine":  "#CCCCCC",
    "grid":   "#E8E8E8",
    "annot":  "#444444",
}

# Convenience aliases matching the old dict keys (stroke layer for lines)
COLORS = {
    "exp_a":      C["exp_a_stroke"],
    "exp_b":      C["exp_b_stroke"],
    "exp_c":      C["exp_c_stroke"],
    "exp_d":      C["exp_d_stroke"],
    "successful": C["succ_stroke"],
    "failure":    C["fail_stroke"],
}

# Custom colormaps
CMAP_PASTEL_BLUE = LinearSegmentedColormap.from_list(
    "pastel_blue", ["#F7FBFD", "#BFDCE7", "#6DAFC2"], N=256)


def setup_style():
    """Apply global publication-quality style with pastel palette."""
    sns.set_style("whitegrid", {
        "grid.color":    C["grid"],
        "grid.linestyle": "-",
        "axes.edgecolor": C["spine"],
    })

    HAS_LATEX = os.system("which pdflatex > /dev/null 2>&1") == 0
    if HAS_LATEX:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["mathtext.fontset"] = "cm"

    FONT_SIZE = 16
    plt.rc("axes",   titlesize=FONT_SIZE, labelsize=FONT_SIZE)
    plt.rc("xtick",  labelsize=FONT_SIZE - 2)
    plt.rc("ytick",  labelsize=FONT_SIZE - 2)
    plt.rc("legend", fontsize=FONT_SIZE - 2)
    plt.rc("figure", titlesize=FONT_SIZE + 2)

    mpl.rcParams["text.color"]       = C["text"]
    mpl.rcParams["axes.labelcolor"]  = C["text"]
    mpl.rcParams["xtick.color"]      = C["text"]
    mpl.rcParams["ytick.color"]      = C["text"]
    mpl.rcParams["axes.titlepad"]    = 10
    mpl.rcParams["grid.alpha"]       = 0.35
    mpl.rcParams["savefig.dpi"]      = 200
    mpl.rcParams["savefig.bbox"]     = "tight"


# Unified font-size tiers for consistency across all figures
# Use these constants instead of ad-hoc numbers.
FS = {
    "suptitle":   16,   # figure-level title
    "ax_title":   13,   # subplot title
    "ax_label":   13,   # x/y axis labels
    "tick":       11,   # tick labels
    "annot":      11,   # value annotations on bars
    "legend":     12,   # legend entries
    "bar_annot":  11,   # bar value text
    # Multi-panel grids (>=5 cols) – slightly smaller
    "grid_title": 11,   # subplot title in dense grids
    "grid_label": 11,   # axis labels in dense grids
    "grid_tick":  10,   # tick labels in dense grids
}


def style_ax(ax, left=True, bottom=True):
    """Minimalist axis: only left+bottom spines in light gray."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["spine"])
        ax.spines[s].set_linewidth(0.8)


setup_style()
FONT_SIZE = 16

# ===================================================================
# Paths
# ===================================================================
ROBOMETER_ROOT = Path("/root/autodl-tmp/robometer")
EVAL_OUT       = ROBOMETER_ROOT / "baseline_eval_output"
EVAL_OUT_2B    = ROBOMETER_ROOT / "baseline_eval_output-2b"
LOGS_ROOT      = ROBOMETER_ROOT / "logs"
RL_EVAL_DIR    = LOGS_ROOT / "rl_eval"
RL_RUNS_DIR    = ROBOMETER_ROOT / "runs"
PROCESSED_DS   = Path("/root/autodl-tmp/processed_datasets")
OUT_DIR        = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ===================================================================
# Utilities
# ===================================================================

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
# Figure 0: Monotonicity Trap Conceptual Diagram
# ===================================================================
def gen_fig0_monotonicity_trap():
    """Two-panel conceptual illustration of the Monotonicity Trap.

    Left : Three potential functions (Linear, Exponential, Sigmoid)
           over states s1..s4 -- all satisfy the same ranking.
    Right: Corresponding per-step increment distributions (bar chart).
    """
    states = np.array([1, 2, 3, 4])
    state_labels = [r"$s_1$", r"$s_2$", r"$s_3$", r"$s_4$"]

    phi_linear = np.array([0.25, 0.50, 0.75, 1.00])
    phi_exp    = np.array([0.01, 0.04, 0.15, 1.00])
    phi_sig    = np.array([0.02, 0.85, 0.95, 1.00])

    curves = [
        ("Linear",      phi_linear, C["exp_c_stroke"], C["exp_c_fill"]),
        ("Exponential",  phi_exp,    C["exp_d_stroke"], C["exp_d_fill"]),
        ("Sigmoid",      phi_sig,    C["exp_b_stroke"], C["exp_b_fill"]),
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.5),
                                             gridspec_kw={"width_ratios": [1, 1.2]})

    # --- Left panel: potential functions ---
    x_smooth = np.linspace(1, 4, 200)
    for label, phi, stroke, fill in curves:
        from scipy.interpolate import PchipInterpolator
        interp = PchipInterpolator(states, phi)
        y_smooth = interp(x_smooth)
        ax_left.plot(x_smooth, y_smooth, color=stroke, linewidth=2.5, label=label, zorder=3)
        ax_left.scatter(states, phi, color=fill, edgecolor=stroke, linewidth=1.5,
                        s=70, zorder=4)

    ax_left.set_xticks(states)
    ax_left.set_xticklabels(state_labels, fontsize=FS["tick"])
    ax_left.set_ylabel(r"$\Phi(s)$", fontsize=FS["ax_label"])
    ax_left.set_xlabel("State", fontsize=FS["ax_label"])
    ax_left.set_title("(a) Potential Functions (Same Ranking)", fontsize=FS["ax_title"], pad=12)
    ax_left.legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])
    ax_left.set_ylim(-0.05, 1.15)
    style_ax(ax_left)

    # --- Right panel: increment distributions ---
    delta_labels = [r"$\Delta\Phi_1$", r"$\Delta\Phi_2$", r"$\Delta\Phi_3$"]
    x_pos = np.arange(len(delta_labels))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    for i, (label, phi, stroke, fill) in enumerate(curves):
        deltas = np.diff(phi)
        bars = ax_right.bar(x_pos + offsets[i], deltas, bar_width * 0.9,
                            label=label, color=fill, edgecolor=stroke, linewidth=1.2)
        for bar, val in zip(bars, deltas):
            ax_right.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.02,
                          f"{val:.2f}", ha="center", va="bottom",
                          fontsize=FS["bar_annot"], color=C["annot"], fontweight="bold")

    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(delta_labels, fontsize=FS["tick"])
    ax_right.set_ylabel(r"$\Delta\Phi = \Phi(s_{i+1}) - \Phi(s_i)$", fontsize=FS["ax_label"])
    ax_right.set_title("(b) Increment Distributions", fontsize=FS["ax_title"], pad=12)
    ax_right.legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])
    ax_right.set_ylim(-0.05, 1.0)
    style_ax(ax_right)

    plt.tight_layout(w_pad=3)
    out_path = OUT_DIR / "monotonicity-trap.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [Fig0] Saved {out_path}")
    return [out_path]


# ===================================================================
# Figure 1: LIBERO Environment Overview
# ===================================================================
def gen_fig1_libero_overview():
    results_file = (EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"
                    / "reward_alignment" / "libero_90_libero_90_failure_results.json")
    data = load_json(results_file)

    successful = sorted(
        [d for d in data if d["quality_label"] == "successful"],
        key=lambda x: x["task"],
    )
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
        axes = axes.flatten() if rows > 1 else axes.flatten()

        loaded_count = 0
        for i, traj in enumerate(selected):
            frames = load_npz_frames(traj["video_path"])
            if frames is None:
                axes[i].text(0.5, 0.5, "N/A", ha="center", va="center",
                             fontsize=FS["ax_label"], color=C["text"])
                axes[i].set_title(traj["task"][:40] + "...", fontsize=FS["grid_title"], pad=4,
                                  color=C["text"])
                axes[i].axis("off")
                continue

            pick_indices = [0, len(frames) // 3, 2 * len(frames) // 3]
            frame_idx = min(pick_indices[min(variant_idx, len(pick_indices) - 1)],
                            len(frames) - 1)
            axes[i].imshow(frames[frame_idx])
            task_short = traj["task"][:45] + "..." if len(traj["task"]) > 45 else traj["task"]
            axes[i].set_title(task_short, fontsize=FS["grid_title"], pad=4, color=C["text"])
            axes[i].axis("off")
            loaded_count += 1

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle("LIBERO-90 Benchmark Tasks (Unseen Evaluation Set)",
                      fontsize=FONT_SIZE, y=1.02, color=C["text"])
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
    results_90 = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"
        / "policy_ranking" / "libero_90_libero_90_failure_task_groups.json"
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
            agg_key = ("final_predicted_reward_avg" if variant_idx == 0
                       else "final_predicted_reward_sum")

            bar_labels = []
            bar_vals = []
            bar_colors_fill = []
            bar_colors_edge = []
            for si, s in enumerate(succ_items[:2]):
                bar_labels.append(f"S{si+1}")
                bar_vals.append(s[agg_key])
                bar_colors_fill.append(C["succ_fill"])
                bar_colors_edge.append(C["succ_stroke"])
            for fi, f_item in enumerate(fail_items[:2]):
                bar_labels.append(f"F{fi+1}")
                bar_vals.append(f_item[agg_key])
                bar_colors_fill.append(C["fail_fill"])
                bar_colors_edge.append(C["fail_stroke"])

            x_pos = np.arange(len(bar_labels))
            bw = 0.45
            ax.bar(x_pos, bar_vals, bw, color=bar_colors_fill,
                   edgecolor=bar_colors_edge, linewidth=1.0, alpha=0.9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, fontsize=FS["tick"])

            task_short = task if len(task) <= 35 else task[:32] + "..."
            ax.set_title(task_short, fontsize=FS["ax_title"], pad=6, color=C["text"])
            ax.set_ylabel("Predicted Reward" if i % cols == 0 else "",
                          fontsize=FS["ax_label"])
            style_ax(ax)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        agg_label = "Average" if variant_idx == 0 else "Sum"
        fig.suptitle(
            f"Ours (BT+MaxEnt): Predicted Rewards \u2014 Successful vs Failed ({agg_label})",
            fontsize=FONT_SIZE, color=C["text"])
        plt.tight_layout()
        out_path = OUT_DIR / f"fig2_reward_succ_vs_fail_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig2 v{variant_idx+1}] Saved {out_path}")

    # Variant 3: per-frame progress curves
    results_ra = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"
        / "policy_ranking" / "libero_90_libero_90_failure_results.json"
    )
    task_to_trajs = defaultdict(list)
    for d in results_ra:
        task_to_trajs[d["task"]].append(d)

    sorted_tasks = sorted(task_to_trajs.keys())
    n_tasks_show = min(10, len(sorted_tasks))
    cols = min(5, n_tasks_show)
    rows = (n_tasks_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    max_trajs_per_label = 5
    global_min = min(v for d in results_ra for v in d["progress_pred"])
    global_max = max(v for d in results_ra for v in d["progress_pred"])
    y_margin = (global_max - global_min) * 0.1
    y_lo, y_hi = global_min - y_margin, global_max + y_margin

    for i, task in enumerate(sorted_tasks[:n_tasks_show]):
        ax = axes[i]
        trajs = task_to_trajs[task]
        succ = [t for t in trajs if t["quality_label"] == "successful"][:max_trajs_per_label]
        fail = [t for t in trajs if t["quality_label"] == "failure"][:max_trajs_per_label]

        for traj in fail:
            pp = traj["progress_pred"]
            ax.plot(range(len(pp)), pp, color=C["fail_stroke"], linewidth=2, alpha=0.7)
        for traj in succ:
            pp = traj["progress_pred"]
            ax.plot(range(len(pp)), pp, color=C["succ_stroke"], linewidth=2, alpha=0.7)

        task_short = task if len(task) <= 30 else task[:27] + "..."
        ax.set_title(task_short, fontsize=FS["grid_title"], pad=3, color=C["text"])
        ax.set_ylim(y_lo, y_hi)
        ax.axhline(y=0, color=C["grid"], linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=FS["grid_tick"])
        style_ax(ax)
        if i >= cols:
            ax.set_xlabel("Frame", fontsize=FS["grid_label"])
        if i % cols == 0:
            ax.set_ylabel("Progress", fontsize=FS["grid_label"])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=C["succ_stroke"], lw=2, label="Successful"),
        plt.Line2D([0], [0], color=C["fail_stroke"], lw=2, label="Failed"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=FS["legend"],
               bbox_to_anchor=(0.5, 1.03), framealpha=0.9, edgecolor=C["spine"])
    fig.suptitle("Ours (BT+MaxEnt): Per-Frame Progress Predictions on LIBERO-90",
                  fontsize=FONT_SIZE, y=1.06, color=C["text"])
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
    results_c = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"
        / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )
    results_d = load_json(
        EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900"
        / "reward_alignment" / "libero_90_libero_90_failure_results.json"
    )
    c_by_id = {d["id"]: d for d in results_c}
    d_by_id = {d["id"]: d for d in results_d}
    common_ids = sorted(set(c_by_id.keys()) & set(d_by_id.keys()))

    all_variants = []
    n_show = min(10, len(common_ids))

    # Compute shared y-range from all displayed trajectories
    all_progress_vals = []
    for tid in common_ids[:n_show]:
        all_progress_vals.extend(c_by_id[tid]["progress_pred"])
        all_progress_vals.extend(d_by_id[tid]["progress_pred"])
        if "target_progress" in c_by_id[tid] and c_by_id[tid]["target_progress"]:
            all_progress_vals.extend(c_by_id[tid]["target_progress"])
    g_ymin = min(all_progress_vals)
    g_ymax = max(all_progress_vals)
    g_ypad = (g_ymax - g_ymin) * 0.10
    g_ylim = (g_ymin - g_ypad, g_ymax + g_ypad)

    # Variant 1: 2x5 grid
    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, tid in enumerate(common_ids[:n_show]):
        ax = axes[i]
        c_traj, d_traj = c_by_id[tid], d_by_id[tid]
        pp_c, pp_d = c_traj["progress_pred"], d_traj["progress_pred"]

        ax.plot(range(len(pp_c)), pp_c, color=C["exp_c_stroke"],
                linewidth=2.5, label="Ours (BT+MaxEnt)")
        ax.plot(range(len(pp_d)), pp_d, color=C["exp_d_stroke"],
                linewidth=2.5, label="Supervised Oracle", linestyle="--")

        if "target_progress" in c_traj and c_traj["target_progress"]:
            tp = c_traj["target_progress"]
            ax.plot(range(len(tp)), tp, color=C["gt"],
                    linewidth=1, linestyle=":", alpha=0.6, label="GT")

        task_short = c_traj["task"]
        if len(task_short) > 30:
            task_short = task_short[:27] + "..."
        ax.set_title(f"{task_short}\n({c_traj['quality_label']})",
                      fontsize=FS["grid_title"], pad=3, color=C["text"])
        ax.set_ylim(*g_ylim)
        ax.tick_params(labelsize=FS["grid_tick"])
        style_ax(ax)
        if i >= cols:
            ax.set_xlabel("Frame", fontsize=FS["grid_label"])
        if i % cols == 0:
            ax.set_ylabel("Progress", fontsize=FS["grid_label"])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=C["exp_c_stroke"], lw=2.5, label="Ours (BT+MaxEnt)"),
        plt.Line2D([0], [0], color=C["exp_d_stroke"], lw=2.5, linestyle="--",
                    label="Supervised Oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=FS["legend"],
               bbox_to_anchor=(0.5, 1.03), framealpha=0.9, edgecolor=C["spine"])
    fig.suptitle("Progress Predictions: Ours vs Supervised Oracle on Same Trajectories",
                  fontsize=FONT_SIZE, y=1.06, color=C["text"])
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
        c_traj, d_traj = c_by_id[tid], d_by_id[tid]
        pp_c, pp_d = c_traj["progress_pred"], d_traj["progress_pred"]

        ax.plot(range(len(pp_c)), pp_c, color=C["exp_c_stroke"],
                linewidth=2.5, label="Ours (BT+MaxEnt)")
        ax.plot(range(len(pp_d)), pp_d, color=C["exp_d_stroke"],
                linewidth=2.5, linestyle="--", label="Supervised Oracle")

        if "target_progress" in c_traj and c_traj["target_progress"]:
            tp = c_traj["target_progress"]
            ax.plot(range(len(tp)), tp, color=C["gt"],
                    linewidth=1, linestyle=":", alpha=0.6, label="GT")

        task_short = c_traj["task"]
        if len(task_short) > 40:
            task_short = task_short[:37] + "..."
        ax.set_title(f"{task_short} ({c_traj['quality_label']})",
                      fontsize=FS["ax_title"], pad=5, color=C["text"])
        ax.set_ylim(*g_ylim)
        ax.set_xlabel("Frame Index", fontsize=FS["ax_label"])
        ax.set_ylabel("Progress" if i % 3 == 0 else "", fontsize=FS["ax_label"])
        ax.tick_params(labelsize=FS["tick"])
        style_ax(ax)

    handles = [
        plt.Line2D([0], [0], color=C["exp_c_stroke"], lw=2.5, label="Ours (BT+MaxEnt)"),
        plt.Line2D([0], [0], color=C["exp_d_stroke"], lw=2.5, linestyle="--",
                    label="Supervised Oracle"),
        plt.Line2D([0], [0], color=C["gt"], lw=1, linestyle=":", alpha=0.6,
                    label="Ground Truth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=FS["legend"],
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9, edgecolor=C["spine"])
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
    metrics_c = load_all_metrics(EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000")
    metrics_d = load_all_metrics(EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900")

    all_variants = []

    # ---- Variant 1: LIBERO-90 metrics (2 methods) ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_configs = [
        ("VOC r (Reward Alignment)", "reward_alignment",
         "libero_90_libero_90_failure/pearson"),
        ("Kendall tau (Policy Ranking)", "policy_ranking",
         "libero_90_libero_90_failure/kendall_sum"),
        ("Ranking Accuracy", "policy_ranking",
         "libero_90_libero_90_failure/ranking_acc_sum"),
        ("Succ-Fail Diff", "policy_ranking",
         "libero_90_libero_90_failure/avg_succ_fail_diff_sum"),
    ]
    methods = ["Ours\n(BT+MaxEnt)", "Supervised\nOracle"]
    method_fills   = [C["exp_c_fill"],   C["exp_d_fill"]]
    method_strokes = [C["exp_c_stroke"], C["exp_d_stroke"]]
    x = np.arange(len(methods))
    width = 0.35

    for ax_idx, (title, section, key) in enumerate(metric_configs):
        ax = axes[ax_idx]
        vals = []
        for m_data in [metrics_c, metrics_d]:
            vals.append(m_data.get(section, {}).get(key, 0))

        bars = ax.bar(x, vals, width, color=method_fills,
                      edgecolor=method_strokes, linewidth=1.2)
        for bar, val in zip(bars, vals):
            y_off = 0.02 * max(abs(v) for v in vals) if vals else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=FS["bar_annot"], fontweight="bold", color=C["annot"])

        ax.set_title(title, fontsize=FS["ax_title"], pad=8, color=C["text"])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=FS["tick"])
        style_ax(ax)

    fig.suptitle("LIBERO-90: Ours (BT+MaxEnt) vs Supervised Oracle",
                  fontsize=FONT_SIZE, y=1.03, color=C["text"])
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
        w = 0.28
        method_meta = [
            (metrics_c, "Ours (BT+MaxEnt)", C["exp_c_fill"], C["exp_c_stroke"]),
            (metrics_d, "Supervised Oracle", C["exp_d_fill"], C["exp_d_stroke"]),
        ]
        for mi, (m_data, m_label, m_fill, m_stroke) in enumerate(method_meta):
            vals = []
            for sec, key in zip(sections, metric_keys):
                vals.append(m_data.get(sec, {}).get(key, 0))
            offset = (mi - 0.5) * w
            bars = ax.bar(x + offset, vals, w * 0.85, label=m_label,
                          color=m_fill, edgecolor=m_stroke, linewidth=1.0, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=FS["bar_annot"], color=C["annot"])

        ax.set_title(dataset_label, fontsize=FS["ax_title"], pad=8, color=C["text"])
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=FS["tick"])
        ax.set_ylim(-0.2, 1.15)
        ax.axhline(y=0, color=C["grid"], linewidth=0.5)
        style_ax(ax)
        if ax_idx == 0:
            ax.legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])

    fig.suptitle("Reward Model Evaluation: Ours vs Supervised Oracle",
                  fontsize=FONT_SIZE, y=1.02, color=C["text"])
    plt.tight_layout()
    out_path = OUT_DIR / "fig4_main_results_bar_v2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig4 v2] Saved {out_path}")

    # ---- Variant 3: All 4 methods (A/B/C/D) ----
    metrics_a = load_all_metrics(
        EVAL_OUT / "rbm_exp_a_pure_bt_smolvlm_checkpoint-1000")
    metrics_b = load_all_metrics(
        EVAL_OUT / "rbm_exp_b_l2_smooth_smolvlm_checkpoint-1000")

    if metrics_a and metrics_b:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        metric_configs_4 = [
            ("VOC r", "reward_alignment",
             "libero_90_libero_90_failure/pearson"),
            ("Kendall tau", "policy_ranking",
             "libero_90_libero_90_failure/kendall_sum"),
            ("Ranking Accuracy", "policy_ranking",
             "libero_90_libero_90_failure/ranking_acc_sum"),
            ("Succ-Fail Diff", "policy_ranking",
             "libero_90_libero_90_failure/avg_succ_fail_diff_sum"),
        ]
        all_methods = [
            ("Pure BT",           metrics_a, C["exp_a_fill"], C["exp_a_stroke"]),
            ("BT+L2\nSmooth",     metrics_b, C["exp_b_fill"], C["exp_b_stroke"]),
            ("Ours\n(BT+MaxEnt)", metrics_c, C["exp_c_fill"], C["exp_c_stroke"]),
            ("Supervised\nOracle", metrics_d, C["exp_d_fill"], C["exp_d_stroke"]),
        ]

        for ax_idx, (title, section, key) in enumerate(metric_configs_4):
            ax = axes[ax_idx]
            x = np.arange(len(all_methods))
            vals, fills, strokes, labels = [], [], [], []
            for label, m_data, fill, stroke in all_methods:
                vals.append(m_data.get(section, {}).get(key, 0))
                fills.append(fill)
                strokes.append(stroke)
                labels.append(label)

            bars = ax.bar(x, vals, 0.45, color=fills,
                          edgecolor=strokes, linewidth=1.2)
            for bar, val in zip(bars, vals):
                y_off = 0.02 * max(abs(v) for v in vals) if vals else 0
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_off,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=FS["bar_annot"], fontweight="bold", color=C["annot"])
            ax.set_title(title, fontsize=FS["ax_title"], pad=8, color=C["text"])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=FS["tick"])
            ax.axhline(y=0, color=C["grid"], linewidth=0.5)
            style_ax(ax)

        fig.suptitle("LIBERO-90: Ablation Study \u2014 Four Methods Compared",
                      fontsize=FONT_SIZE, y=1.03, color=C["text"])
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
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    all_variants = []

    exp_configs = {
        "Ours (BT+MaxEnt)": {
            "tb_dir": str(LOGS_ROOT / "exp_c_dirfix_v1" / "exp_c_dirfix_v1" / "tb"),
            "stroke": C["exp_c_stroke"],
            "fill":   C["exp_c_fill"],
            "tags": {
                "Preference Loss": "train/preference_loss",
                "Struct Loss":     "train/struct_loss",
                "Preference Accuracy": "train/preference_accuracy",
                "Delta Variance":  "train/delta_variance",
                "Delta Mean":      "train/delta_mean",
            },
        },
        "Supervised Oracle (RoboMeter)": {
            "tb_dir": str(LOGS_ROOT / "exp_d_robometer_smolvlm"
                          / "exp_d_robometer_smolvlm" / "tb"),
            "stroke": C["exp_d_stroke"],
            "fill":   C["exp_d_fill"],
            "tags": {
                "Preference Loss": "train/preference_loss",
                "Preference Accuracy": "train/preference_accuracy",
                "Progress Loss":   "train/pref_prog_loss",
                "Progress Spearman": "train/pref_prog_spearman_corr",
            },
        },
    }

    all_data = {}
    for exp_name, config in exp_configs.items():
        ea = EventAccumulator(config["tb_dir"])
        ea.Reload()
        exp_data = {}
        for tag_label, tag_name in config["tags"].items():
            try:
                events = ea.Scalars(tag_name)
                exp_data[tag_label] = ([e.step for e in events],
                                       [e.value for e in events])
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
            ax.plot(steps, values, color=C["exp_c_fill"], linewidth=0.8)
            smoothed = smooth(values, weight=0.8)
            ax.plot(steps, smoothed, color=C["exp_c_stroke"], linewidth=2.5,
                    label="Ours")
        ax.set_title(tag, fontsize=FS["ax_title"], pad=6, color=C["text"])
        ax.set_xlabel("Training Step", fontsize=FS["ax_label"])
        ax.set_ylabel(ylabel, fontsize=FS["ax_label"])
        ax.tick_params(labelsize=FS["tick"])
        style_ax(ax)

    if "Struct Loss" in c_data:
        axes[1].axhline(y=-np.log(7), color=C["fail_stroke"], linestyle="--",
                        linewidth=1, alpha=0.6,
                        label=r"Saturation: $-\ln(7)$")
        axes[1].legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])

    fig.suptitle("Ours (BT+MaxEnt) Training Dynamics (SmolVLM-500M)",
                  fontsize=FONT_SIZE, y=1.01, color=C["text"])
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
    d_tags = ["Preference Loss", "Preference Accuracy",
              "Progress Loss", "Progress Spearman"]
    d_ylabels = ["Loss", "Accuracy", "Loss", "Spearman Corr"]

    for i, (tag, ylabel) in enumerate(zip(d_tags, d_ylabels)):
        ax = axes[i]
        if tag in d_data:
            steps, values = d_data[tag]
            ax.plot(steps, values, color=C["exp_d_fill"], linewidth=0.8)
            smoothed = smooth(values, weight=0.8)
            ax.plot(steps, smoothed, color=C["exp_d_stroke"], linewidth=2.5)
        ax.set_title(tag, fontsize=FS["ax_title"], pad=6, color=C["text"])
        ax.set_xlabel("Training Step", fontsize=FS["ax_label"])
        ax.set_ylabel(ylabel, fontsize=FS["ax_label"])
        ax.tick_params(labelsize=FS["tick"])
        style_ax(ax)

    fig.suptitle("Supervised Oracle Training Dynamics (SmolVLM-500M)",
                  fontsize=FONT_SIZE, y=1.01, color=C["text"])
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v2] Saved {out_path}")

    # ---- Variant 3: Combined — Exp C vs D overlaid ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, tag in enumerate(["Preference Loss", "Preference Accuracy"]):
        ax = axes[ax_idx]
        for exp_name, exp_data in all_data.items():
            if tag in exp_data:
                steps, values = exp_data[tag]
                stroke = exp_configs[exp_name]["stroke"]
                fill   = exp_configs[exp_name]["fill"]
                ax.plot(steps, values, color=fill, linewidth=0.6)
                smoothed = smooth(values, weight=0.8)
                ax.plot(steps, smoothed, color=stroke, linewidth=2.5, label=exp_name)
        ax.set_title(tag, fontsize=FS["ax_title"], pad=6, color=C["text"])
        ax.set_xlabel("Training Step", fontsize=FS["ax_label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])
        style_ax(ax)

    fig.suptitle("Training Comparison: Ours vs Supervised Oracle (SmolVLM-500M)",
                  fontsize=FONT_SIZE, y=1.02, color=C["text"])
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v3.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v3] Saved {out_path}")

    # ---- Variant 4: Dual y-axis (BT loss + struct loss) ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    if "Preference Loss" in c_data:
        steps, values = c_data["Preference Loss"]
        smoothed = smooth(values, weight=0.85)
        ax1.plot(steps, smoothed, color=C["exp_c_stroke"], linewidth=2.5,
                 label=r"$\mathcal{L}_{BT}$")
        ax1.plot(steps, values, color=C["exp_c_fill"], linewidth=0.6)

    if "Struct Loss" in c_data:
        steps, values = c_data["Struct Loss"]
        smoothed = smooth(values, weight=0.85)
        ax2.plot(steps, smoothed, color=C["struct_stroke"], linewidth=2.5,
                 label=r"$\mathcal{L}_{struct}$")
        ax2.plot(steps, values, color=C["struct_fill"], linewidth=0.6)
        ax2.axhline(y=-np.log(7), color=C["fail_stroke"], linestyle="--",
                     linewidth=1, alpha=0.5)
        ax2.text(steps[-1] * 0.7, -np.log(7) + 0.03,
                 r"Saturation: $-\ln(7)$", fontsize=FS["tick"],
                 color=C["fail_stroke"], alpha=0.8)

    ax1.set_xlabel("Training Step", fontsize=FS["ax_label"])
    ax1.set_ylabel(r"$\mathcal{L}_{BT}$ (Preference Loss)", fontsize=FS["ax_label"],
                    color=C["exp_c_stroke"])
    ax1.tick_params(labelsize=FS["tick"])
    ax2.set_ylabel(r"$\mathcal{L}_{struct}$ (Entropy Loss)", fontsize=FS["ax_label"],
                    color=C["struct_stroke"])
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    for s in ("left", "bottom", "right"):
        ax1.spines[s].set_color(C["spine"])
        ax1.spines[s].set_linewidth(0.8)
    ax2.spines["right"].set_color(C["spine"])
    ax2.spines["right"].set_linewidth(0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.tick_params(labelsize=FS["tick"])
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=FS["legend"], loc="upper right",
               framealpha=0.9, edgecolor=C["spine"])

    fig.suptitle("Ours (BT+MaxEnt): Joint Loss Training Dynamics",
                  fontsize=FONT_SIZE, y=1.01, color=C["text"])
    plt.tight_layout()
    out_path = OUT_DIR / "fig5_training_dynamics_v4_dual_axis.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    all_variants.append(out_path)
    print(f"  [Fig5 v4] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 6: Confusion Matrix Heatmap + Reward Separation
# ===================================================================
def gen_fig6_confusion_matrix():
    all_variants = []

    for variant_idx, (exp_label, exp_dir, stroke_color) in enumerate([
        ("Ours (BT+MaxEnt)",
         EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000",
         C["exp_c_stroke"]),
        ("Supervised Oracle",
         EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900",
         C["exp_d_stroke"]),
    ]):
        task_groups_file = (exp_dir / "policy_ranking"
                            / "libero_90_libero_90_failure_task_groups.json")
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
            matrix = np.divide(matrix, count_matrix,
                               out=np.zeros_like(matrix),
                               where=count_matrix != 0)

        fig, ax = plt.subplots(figsize=(6, 5))
        task_labels_short = [t[:25] + "..." if len(t) > 25 else t for t in tasks]

        sns.heatmap(matrix, annot=True, fmt=".3f", cmap=CMAP_PASTEL_BLUE,
                    ax=ax, xticklabels=task_labels_short,
                    yticklabels=task_labels_short,
                    cbar_kws={"label": "Avg Predicted Reward"},
                    linewidths=0.5, linecolor=C["grid"])
        ax.set_xlabel("Task", fontsize=FS["ax_label"])
        ax.set_ylabel("Task", fontsize=FS["ax_label"])
        plt.xticks(rotation=45, ha="right", fontsize=FS["grid_tick"])
        plt.yticks(rotation=0, fontsize=FS["grid_tick"])

        fig.suptitle(f"Reward Heatmap: {exp_label}", fontsize=FS["ax_title"], y=1.02,
                      color=C["text"])
        plt.tight_layout()
        out_path = OUT_DIR / f"fig6_confusion_matrix_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig6 v{variant_idx+1}] Saved {out_path}")

    # Variant 3: Side-by-side reward separation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (exp_label, exp_dir) in enumerate([
        ("Ours (BT+MaxEnt)",
         EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"),
        ("Supervised Oracle",
         EVAL_OUT / "rbm_exp_d_robometer_smolvlm_checkpoint-900"),
    ]):
        ax = axes[ax_idx]
        task_groups_file = (exp_dir / "policy_ranking"
                            / "libero_90_libero_90_failure_task_groups.json")
        if not task_groups_file.exists():
            continue

        task_groups = load_json(task_groups_file)
        tasks = sorted(task_groups.keys())

        succ_rewards, fail_rewards, task_labels = [], [], []
        for task in tasks:
            items = task_groups[task]
            s_vals = [it["final_predicted_reward_avg"]
                      for it in items if it["quality_label"] == "successful"]
            f_vals = [it["final_predicted_reward_avg"]
                      for it in items if it["quality_label"] == "failure"]
            if s_vals and f_vals:
                succ_rewards.append(np.mean(s_vals))
                fail_rewards.append(np.mean(f_vals))
                task_labels.append(task[:20] + "..." if len(task) > 20 else task)

        x = np.arange(len(task_labels))
        w = 0.30
        ax.bar(x - w / 2, succ_rewards, w, label="Successful",
               color=C["succ_fill"], edgecolor=C["succ_stroke"], linewidth=0.8)
        ax.bar(x + w / 2, fail_rewards, w, label="Failed",
               color=C["fail_fill"], edgecolor=C["fail_stroke"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha="right",
                           fontsize=FS["grid_tick"])
        ax.set_title(exp_label, fontsize=FS["ax_title"], color=C["text"])
        ax.set_ylabel("Avg Predicted Reward", fontsize=FS["ax_label"])
        ax.legend(fontsize=FS["legend"], framealpha=0.9, edgecolor=C["spine"])
        style_ax(ax)

    fig.suptitle("Reward Separation: Successful vs Failed per Task",
                  fontsize=FONT_SIZE, y=1.03, color=C["text"])
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
    results_c = load_json(
        EVAL_OUT / "rbm_exp_c_dirfix_v1_checkpoint-1000"
        / "reward_alignment" / "libero_90_libero_90_failure_results.json"
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

        # Compute shared y-range from ALL selected trajectories in this variant
        all_vals = []
        for traj in selected:
            all_vals.extend(traj["progress_pred"])
            if "target_progress" in traj and traj["target_progress"]:
                all_vals.extend(traj["target_progress"])
        y_min = min(all_vals)
        y_max = max(all_vals)
        y_pad = (y_max - y_min) * 0.12
        y_lo = y_min - y_pad
        y_hi = y_max + y_pad

        fig = plt.figure(figsize=(16, 4 * n_trajs))
        gs = GridSpec(n_trajs, 5, figure=fig, width_ratios=[1, 1, 1, 1, 2])

        for row, traj in enumerate(selected):
            frames = load_npz_frames(traj["video_path"])
            progress = traj["progress_pred"]
            task = traj["task"]
            ql = traj["quality_label"]

            if frames is not None:
                n_frames = len(frames)
                pick = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
                for col, fi in enumerate(pick):
                    ax = fig.add_subplot(gs[row, col])
                    ax.imshow(frames[fi])
                    ax.set_title(f"Frame {fi}", fontsize=FS["grid_title"], color=C["text"])
                    ax.axis("off")
            else:
                for col in range(4):
                    ax = fig.add_subplot(gs[row, col])
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            fontsize=FS["ax_label"], color=C["text"])
                    ax.axis("off")

            ax_plot = fig.add_subplot(gs[row, 4])
            stroke = C["succ_stroke"] if ql == "successful" else C["fail_stroke"]
            ax_plot.plot(range(len(progress)), progress,
                         color=stroke, linewidth=2.5)
            ax_plot.set_ylim(y_lo, y_hi)
            ax_plot.set_ylabel("Progress", fontsize=FS["ax_label"])
            ax_plot.set_xlabel("Frame" if row == n_trajs - 1 else "",
                               fontsize=FS["ax_label"])
            ax_plot.tick_params(labelsize=FS["tick"])
            task_short = task if len(task) <= 40 else task[:37] + "..."
            ax_plot.set_title(f"{task_short} ({ql})", fontsize=FS["ax_title"], pad=4,
                               color=C["text"])
            style_ax(ax_plot)

            if "target_progress" in traj and traj["target_progress"]:
                tp = traj["target_progress"]
                ax_plot.plot(range(len(tp)), tp, color=C["gt"],
                             linewidth=1, linestyle=":", alpha=0.6)

        fig.suptitle("Trajectory Frames and Progress Predictions (Ours)",
                      fontsize=FONT_SIZE, y=1.01, color=C["text"])
        plt.tight_layout()
        out_path = OUT_DIR / f"fig7_trajectory_frames_v{variant_idx + 1}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig7 v{variant_idx+1}] Saved {out_path}")

    return all_variants


# ===================================================================
# Figure 8: RL Success Rate Curve
# ===================================================================
def gen_fig8_rl_success_rate():
    """RL learning curves from TensorBoard event files or JSON logs."""
    all_variants = []

    # Try TensorBoard events first (richer data)
    tb_runs = {}
    if RL_RUNS_DIR.exists():
        for name in sorted(os.listdir(RL_RUNS_DIR)):
            full = RL_RUNS_DIR / name
            if not full.is_dir():
                continue
            has_events = any(f.startswith("events.out.tfevents")
                             for f in os.listdir(full))
            if has_events:
                if "sparse" in name.lower():
                    tb_runs.setdefault("Sparse Reward", []).append(str(full))
                else:
                    tb_runs.setdefault("BT+MaxEnt (Ours)", []).append(str(full))

    if tb_runs:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        legend_entries = []

        palette = {
            "BT+MaxEnt (Ours)": (C["exp_c_stroke"], C["exp_c_fill"]),
            "Sparse Reward":    (C["exp_a_stroke"], C["exp_a_fill"]),
        }

        for run_label in ["BT+MaxEnt (Ours)", "Sparse Reward"]:
            if run_label not in tb_runs:
                continue
            stroke, fill = palette[run_label]

            all_steps, all_values = [], []
            for run_dir in tb_runs[run_label]:
                ea = EventAccumulator(run_dir)
                ea.Reload()
                tag = "eval/success_rate"
                if tag not in ea.Tags().get("scalars", []):
                    for candidate in ea.Tags().get("scalars", []):
                        if "success" in candidate.lower():
                            tag = candidate
                            break
                    else:
                        continue
                events = ea.Scalars(tag)
                steps  = [e.step for e in events]
                values = [e.value for e in events]
                if steps:
                    all_steps.append(steps)
                    all_values.append(values)

            if not all_steps:
                continue

            for steps, values in zip(all_steps, all_values):
                smoothed = smooth(values, weight=0.6)
                ax.plot(steps, values, color=fill, linewidth=0.7, alpha=0.5)
                line, = ax.plot(steps, smoothed, color=stroke, linewidth=2.5)

            legend_entries.append(
                plt.Line2D([0], [0], color=stroke, lw=2.5, label=run_label))

        ax.set_xlabel("Training Steps", fontsize=FS["ax_label"])
        ax.set_ylabel("Eval Success Rate", fontsize=FS["ax_label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_ylim(-0.02, 1.05)
        ax.legend(handles=legend_entries, fontsize=FS["legend"], framealpha=0.9,
                  edgecolor=C["spine"])
        style_ax(ax)

        fig.suptitle("Policy Learning: BT+MaxEnt vs Sparse Reward (Task 28)",
                      fontsize=FONT_SIZE, y=1.01, color=C["text"])
        plt.tight_layout()
        out_path = OUT_DIR / "rl_success_rate.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        all_variants.append(out_path)
        print(f"  [Fig8] Saved {out_path}")

    # Fallback: JSON logs
    if not all_variants and RL_EVAL_DIR.exists():
        fig, ax = plt.subplots(figsize=(8, 5))
        json_configs = [
            ("BT+MaxEnt (Ours)", "rl_exp_c_entropy_seed0.json",
             C["exp_c_stroke"], C["exp_c_fill"]),
            ("Sparse Reward", "rl_exp_d_robometer_seed42.json",
             C["exp_a_stroke"], C["exp_a_fill"]),
        ]
        legend_entries = []
        for label, fname, stroke, fill in json_configs:
            fpath = RL_EVAL_DIR / fname
            if not fpath.exists():
                continue
            records = load_json(fpath)
            steps  = [r["timestep"] for r in records]
            values = [r["success_rate"] for r in records]
            smoothed = smooth(values, weight=0.4)
            ax.plot(steps, values, color=fill, linewidth=0.7, alpha=0.5)
            ax.plot(steps, smoothed, color=stroke, linewidth=2.5)
            legend_entries.append(
                plt.Line2D([0], [0], color=stroke, lw=2.5, label=label))

        if legend_entries:
            ax.set_xlabel("Training Steps", fontsize=FS["ax_label"])
            ax.set_ylabel("Eval Success Rate", fontsize=FS["ax_label"])
            ax.tick_params(labelsize=FS["tick"])
            ax.set_ylim(-0.02, 1.05)
            ax.legend(handles=legend_entries, fontsize=FS["legend"], framealpha=0.9,
                      edgecolor=C["spine"])
            style_ax(ax)
            fig.suptitle("Policy Learning: BT+MaxEnt vs Sparse Reward (Task 28)",
                          fontsize=FONT_SIZE, y=1.01, color=C["text"])
            plt.tight_layout()
            out_path = OUT_DIR / "rl_success_rate.pdf"
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            all_variants.append(out_path)
            print(f"  [Fig8 fallback] Saved {out_path}")
        plt.close(fig)

    if not all_variants:
        print("  [Fig8] Skipped — no RL data found")

    return all_variants


# ===================================================================
# Figure 8-target: RL Learning Curves (from manual eval data)
# ===================================================================
def gen_fig8_rl_target():
    """
    Generate RL training curves with shaded seed-spread.
    Data: 5 seeds x N eval checkpoints (every 20k steps).
    Replace the placeholder arrays below with real success rates.
    """

    # --- Eval checkpoints (training steps) ---
    steps = np.array([0, 20_000, 40_000, 60_000, 80_000])

    # --- Success rates per seed  [seed0 … seed4] ---
    # >>>  REPLACE with real data  <<<
    sparse_seeds = np.array([
        [0.00, 0.00, 0.10, 0.50, 0.55],   # seed 0
        [0.00, 0.10, 0.15, 0.45, 0.70],   # seed 1
        [0.00, 0.00, 0.35, 0.55, 0.75],   # seed 2
        [0.00, 0.00, 0.20, 0.40, 0.60],   # seed 3
        [0.00, 0.00, 0.28, 0.48, 0.44],   # seed 4
    ])

    bt_maxent_seeds = np.array([
        [0.00, 0.00, 0.40, 0.50, 0.70],   # seed 0
        [0.00, 0.15, 0.15, 0.60, 1.00],   # seed 1
        [0.00, 0.06, 0.55, 0.80, 0.95],   # seed 2
        [0.00, 0.10, 0.48, 0.72, 0.70],   # seed 3
        [0.00, 0.15, 0.50, 0.78, 1.00],   # seed 4
    ])

    # --- Plotting config ---
    configs = [
        {
            "label": "Sparse Reward",
            "data": sparse_seeds,
            "stroke": "#9E9E9E",
            "fill": "#D5D5D5",
        },
        {
            "label": "BT+Ent",
            "data": bt_maxent_seeds,
            "stroke": "#6DAFC2",
            "fill": "#BFDCE7",
        },
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cfg in configs:
        data = cfg["data"]                  # (n_seeds, n_steps)
        mean = data.mean(axis=0)
        lo   = data.min(axis=0)
        hi   = data.max(axis=0)

        ax.fill_between(steps, lo, hi,
                         color=cfg["fill"], alpha=0.45)
        ax.plot(steps, mean, color=cfg["stroke"], linewidth=2.5,
                label=cfg["label"], marker='o', markersize=4)

    ax.set_xlabel("Training Steps", fontsize=FS["ax_label"])
    ax.set_ylabel("Eval Success Rate", fontsize=FS["ax_label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.set_xlim(-2000, steps[-1] + 2000)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{int(s // 1000)}k" if s > 0 else "0"
                         for s in steps])

    ax.legend(fontsize=FS["legend"], framealpha=0.9,
              edgecolor=C["spine"], loc="upper left")
    style_ax(ax)
    plt.tight_layout()

    out_path = OUT_DIR / "rl_target_curve.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)

    out_png = OUT_DIR / "rl_target_curve.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)

    plt.close(fig)
    print(f"  [Fig8-target] Saved {out_path}")
    print(f"  [Fig8-target] Saved {out_png}")
    return [out_path, out_png]


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Generating thesis experiment figures (Pastel/Morandi palette)")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    results = {}

    print("\n--- Figure 0: Monotonicity Trap Conceptual Diagram ---")
    results["fig0"] = gen_fig0_monotonicity_trap()

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

    print("\n--- Figure 8: RL Success Rate ---")
    results["fig8"] = gen_fig8_rl_success_rate()

    print("\n--- Figure 8-target: RL Target Curve (manual eval) ---")
    results["fig8t"] = gen_fig8_rl_target()

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
