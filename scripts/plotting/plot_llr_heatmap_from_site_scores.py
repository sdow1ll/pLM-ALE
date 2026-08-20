#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def mutation_position(mutation):
    m = re.match(r"^[A-Z]([0-9]+)[A-Z]$", str(mutation))
    return int(m.group(1)) if m else 10**9


def token_match(col, token):
    parts = re.split(r"[^A-Za-z0-9]+", col)
    return token in parts


def find_column(columns, candidates):
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def find_logp_col(columns, aa, state):
    candidates = [
        f"logp_{aa}_{state}",
        f"log_p_{aa}_{state}",
        f"logprob_{aa}_{state}",
        f"log_prob_{aa}_{state}",
        f"{aa}_logp_{state}",
        f"{aa}_log_p_{state}",
        f"{aa}_logprob_{state}",
        f"{aa}_log_prob_{state}",
        f"{state}_logp_{aa}",
        f"{state}_log_p_{aa}",
        f"{state}_logprob_{aa}",
        f"{state}_log_prob_{aa}",
    ]

    found = find_column(columns, candidates)
    if found:
        return found

    # Robust fallback for slightly different naming.
    for c in columns:
        lc = c.lower()
        if state.lower() not in lc:
            continue
        if not any(x in lc for x in ["logp", "log_p", "logprob", "log_prob"]):
            continue
        if token_match(c, aa):
            return c

    return None


def find_llr_col(columns, aa, state):
    candidates = [
        f"llr_{aa}_{state}",
        f"llr_{aa}_minus_wt_{state}",
        f"llr_{aa}_vs_wt_{state}",
        f"{aa}_llr_{state}",
        f"{state}_llr_{aa}",
        f"{state}_llr_{aa}_minus_wt",
        f"{state}_llr_{aa}_vs_wt",
    ]

    found = find_column(columns, candidates)
    if found:
        return found

    for c in columns:
        lc = c.lower()
        if state.lower() not in lc:
            continue
        if "llr" not in lc:
            continue
        if token_match(c, aa):
            return c

    return None


def compute_llr_values(df, aa, wt, state):
    """
    Return a per-row LLR vector for candidate amino acid aa relative to WT.
    Prefer explicit LLR columns if present; otherwise compute from logp columns.
    """
    columns = list(df.columns)

    if aa == wt:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    llr_col = find_llr_col(columns, aa, state)
    if llr_col is not None:
        return pd.to_numeric(df[llr_col], errors="coerce")

    aa_logp_col = find_logp_col(columns, aa, state)
    wt_logp_col = find_logp_col(columns, wt, state)

    if aa_logp_col is not None and wt_logp_col is not None:
        return (
            pd.to_numeric(df[aa_logp_col], errors="coerce")
            - pd.to_numeric(df[wt_logp_col], errors="coerce")
        )

    helpful_cols = [
        c for c in columns
        if ("log" in c.lower() or "llr" in c.lower())
    ]

    raise ValueError(
        f"Could not find LLR/log-probability columns for aa={aa}, wt={wt}, state={state}.\n"
        f"Some available log/LLR-like columns are:\n{helpful_cols[:80]}"
    )


def build_matrix(df, state):
    rows = []

    group_cols = ["dataset", "gene", "mutation"]
    for needed in group_cols + ["wt", "alt"]:
        if needed not in df.columns:
            raise ValueError(f"Missing required column: {needed}")

    groups = []
    for keys, sub in df.groupby(group_cols, sort=False):
        dataset, gene, mutation = keys
        wt = str(sub["wt"].iloc[0])
        alt = str(sub["alt"].iloc[0])
        label = f"{gene} {mutation}"
        groups.append((dataset, gene, mutation, wt, alt, label, sub.copy()))

    groups.sort(key=lambda x: (x[0], x[1], mutation_position(x[2]), x[2]))

    matrix = []
    labels = []
    wt_list = []
    alt_list = []

    for dataset, gene, mutation, wt, alt, label, sub in groups:
        vals = []
        for aa in AA_LIST:
            llr = compute_llr_values(sub, aa=aa, wt=wt, state=state)
            vals.append(float(np.nanmean(llr.values)))

        matrix.append(vals)
        labels.append(label)
        wt_list.append(wt)
        alt_list.append(alt)

    return np.array(matrix, dtype=float), labels, wt_list, alt_list


def save_heatmap(matrix, labels, alt_list, out_prefix, suffix, title, cmap="coolwarm"):
    out_prefix = Path(out_prefix)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Matrix contains no finite values.")

    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs == 0 or not math.isfinite(max_abs):
        max_abs = 1.0

    fig_width = max(8.0, 0.42 * len(AA_LIST) + 2.8)
    fig_height = max(3.4, 0.48 * len(labels) + 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(np.arange(len(AA_LIST)))
    ax.set_xticklabels(AA_LIST)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Candidate amino acid")
    ax.set_ylabel("Mutation")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean LLR relative to WT" if suffix != "delta_llr_heatmap" else "Mean ΔLLR")

    # Put the actual numeric value in the true ALE mutant residue cell.
    # This replaces the old asterisk marker.
    for row_idx, alt in enumerate(alt_list):
        if alt not in AA_LIST:
            continue
        col_idx = AA_LIST.index(alt)
        value = matrix[row_idx, col_idx]

        if np.isfinite(value):
            label = f"{value:.2f}"
        else:
            label = "NA"

        ax.text(
            col_idx,
            row_idx,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
        )

    fig.tight_layout()

    png = out_prefix.with_name(out_prefix.name + f"_{suffix}.png")
    pdf = out_prefix.with_name(out_prefix.name + f"_{suffix}.pdf")

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_full_csv", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--dataset_filter", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.comparison_full_csv)

    if args.dataset_filter is not None:
        df = df[df["dataset"] == args.dataset_filter].copy()

    if df.empty:
        raise ValueError("No rows after filtering.")

    pretrained, labels, wt_list, alt_list = build_matrix(df, state="pretrained")
    finetuned, labels2, wt_list2, alt_list2 = build_matrix(df, state="finetuned")

    if labels != labels2:
        raise ValueError("Pretrained and fine-tuned row labels do not match.")

    delta = finetuned - pretrained

    save_heatmap(
        pretrained,
        labels,
        alt_list,
        args.out_prefix,
        "pretrained_llr_heatmap",
        "Pretrained mean amino-acid LLR at ALE sites",
    )

    save_heatmap(
        finetuned,
        labels,
        alt_list,
        args.out_prefix,
        "finetuned_llr_heatmap",
        "Fine-tuned mean amino-acid LLR at ALE sites",
    )

    save_heatmap(
        delta,
        labels,
        alt_list,
        args.out_prefix,
        "delta_llr_heatmap",
        "Fine-tuning ΔLLR at ALE sites",
    )


if __name__ == "__main__":
    main()
