#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def mutation_position(mutation):
    m = re.match(r"^[A-Z]([0-9]+)[A-Z]$", str(mutation))
    return int(m.group(1)) if m else 10**9


def add_label_column(df):
    df = df.copy()
    df["mutation_label"] = df["gene"].astype(str) + " " + df["mutation"].astype(str)
    return df


def sort_summary(df):
    df = df.copy()
    df["mutation_pos"] = df["mutation"].apply(mutation_position)
    return df.sort_values(["dataset", "gene", "mutation_pos", "mutation"]).reset_index(drop=True)


def save_current_figure(out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}")


def plot_pretrained_vs_finetuned_mean_llr(summary, out_prefix, title=None):
    summary = sort_summary(add_label_column(summary))

    labels = summary["mutation_label"].tolist()
    x = list(range(len(summary)))
    width = 0.38

    plt.figure(figsize=(max(6.5, 1.4 * len(summary)), 4.8))

    plt.bar(
        [i - width / 2 for i in x],
        summary["mean_pretrained_llr"],
        width=width,
        label="Pretrained",
    )

    plt.bar(
        [i + width / 2 for i in x],
        summary["mean_finetuned_llr"],
        width=width,
        label="Fine-tuned",
    )

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean LLR: log P(mutant) - log P(WT)")

    if title is None:
        title = "Pretrained vs fine-tuned ALE-site preference"

    plt.title(title)
    plt.legend(frameon=False)

    save_current_figure(f"{out_prefix}_main_pretrained_vs_finetuned_mean_llr.png")

    plt.figure(figsize=(max(6.5, 1.4 * len(summary)), 4.8))

    plt.bar(
        [i - width / 2 for i in x],
        summary["mean_pretrained_llr"],
        width=width,
        label="Pretrained",
    )

    plt.bar(
        [i + width / 2 for i in x],
        summary["mean_finetuned_llr"],
        width=width,
        label="Fine-tuned",
    )

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean LLR: log P(mutant) - log P(WT)")
    plt.title(title)
    plt.legend(frameon=False)

    save_current_figure(f"{out_prefix}_main_pretrained_vs_finetuned_mean_llr.pdf")


def plot_mean_delta_llr(summary, out_prefix, title=None):
    summary = sort_summary(add_label_column(summary))

    labels = summary["mutation_label"].tolist()
    x = list(range(len(summary)))

    plt.figure(figsize=(max(6.5, 1.2 * len(summary)), 4.8))
    plt.bar(x, summary["mean_delta_llr"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean ΔLLR: fine-tuned - pretrained")

    if title is None:
        title = "Fine-tuning shift toward ALE mutant residue"

    plt.title(title)

    save_current_figure(f"{out_prefix}_supporting_mean_delta_llr.png")

    plt.figure(figsize=(max(6.5, 1.2 * len(summary)), 4.8))
    plt.bar(x, summary["mean_delta_llr"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean ΔLLR: fine-tuned - pretrained")
    plt.title(title)

    save_current_figure(f"{out_prefix}_supporting_mean_delta_llr.pdf")


def plot_per_sequence_delta_distributions(full, out_prefix, title=None):
    full = add_label_column(full)

    if "delta_llr_finetuned_minus_pretrained" not in full.columns:
        raise ValueError(
            "Full CSV must contain delta_llr_finetuned_minus_pretrained. "
            "Use compare_pretrained_vs_finetuned_site_scores.py first."
        )

    order_df = (
        full[["dataset", "gene", "mutation", "mutation_label"]]
        .drop_duplicates()
        .copy()
    )
    order_df["mutation_pos"] = order_df["mutation"].apply(mutation_position)
    order_df = order_df.sort_values(["dataset", "gene", "mutation_pos", "mutation"])

    labels = order_df["mutation_label"].tolist()

    data = [
        full.loc[
            full["mutation_label"] == label,
            "delta_llr_finetuned_minus_pretrained"
        ].dropna().values
        for label in labels
    ]

    plt.figure(figsize=(max(6.5, 1.2 * len(labels)), 4.8))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Per-sequence ΔLLR: fine-tuned - pretrained")

    if title is None:
        title = "Per-sequence fine-tuning shifts across held-out homologs"

    plt.title(title)

    save_current_figure(f"{out_prefix}_supplement_per_sequence_delta_llr_distribution.png")

    plt.figure(figsize=(max(6.5, 1.2 * len(labels)), 4.8))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Per-sequence ΔLLR: fine-tuned - pretrained")
    plt.title(title)

    save_current_figure(f"{out_prefix}_supplement_per_sequence_delta_llr_distribution.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALE-site LLR figures from pretrained-vs-finetuned comparison CSVs."
    )

    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--full_csv", required=True)
    parser.add_argument("--out_prefix", required=True)

    parser.add_argument(
        "--title_prefix",
        default=None,
        help="Optional prefix for plot titles, e.g. 'ProGen2 E. coli'.",
    )

    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    full = pd.read_csv(args.full_csv)

    required_summary_cols = [
        "dataset",
        "gene",
        "mutation",
        "mean_pretrained_llr",
        "mean_finetuned_llr",
        "mean_delta_llr",
    ]

    missing = [c for c in required_summary_cols if c not in summary.columns]
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {missing}")

    if args.title_prefix:
        main_title = f"{args.title_prefix}: pretrained vs fine-tuned ALE-site preference"
        delta_title = f"{args.title_prefix}: fine-tuning shift toward ALE mutant residue"
        dist_title = f"{args.title_prefix}: per-sequence shifts across held-out homologs"
    else:
        main_title = None
        delta_title = None
        dist_title = None

    plot_pretrained_vs_finetuned_mean_llr(
        summary=summary,
        out_prefix=args.out_prefix,
        title=main_title,
    )

    plot_mean_delta_llr(
        summary=summary,
        out_prefix=args.out_prefix,
        title=delta_title,
    )

    plot_per_sequence_delta_distributions(
        full=full,
        out_prefix=args.out_prefix,
        title=dist_title,
    )


if __name__ == "__main__":
    main()
