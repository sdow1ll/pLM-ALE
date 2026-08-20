#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compare pretrained vs fine-tuned ALE-site scoring outputs."
    )
    parser.add_argument("--pretrained_csv", required=True)
    parser.add_argument("--finetuned_csv", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--comparison_label", default=None)
    args = parser.parse_args()

    pre = pd.read_csv(args.pretrained_csv)
    fin = pd.read_csv(args.finetuned_csv)

    key_cols = [
        "dataset",
        "split",
        "gene",
        "accession",
        "mutation",
        "wt",
        "alt",
        "reference_pos_1based",
        "homolog_mapped_pos_1based",
        "cluster",
    ]

    missing_pre = [c for c in key_cols if c not in pre.columns]
    missing_fin = [c for c in key_cols if c not in fin.columns]

    if missing_pre:
        raise ValueError(f"Missing key columns from pretrained CSV: {missing_pre}")
    if missing_fin:
        raise ValueError(f"Missing key columns from finetuned CSV: {missing_fin}")

    merged = pre.merge(
        fin,
        on=key_cols,
        suffixes=("_pretrained", "_finetuned"),
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError("Merge produced 0 rows. Check that pretrained and fine-tuned CSVs scored the same dataset.")

    merged["delta_llr_finetuned_minus_pretrained"] = (
        merged["llr_alt_minus_wt_finetuned"]
        - merged["llr_alt_minus_wt_pretrained"]
    )

    merged["delta_logp_alt_finetuned_minus_pretrained"] = (
        merged["logp_alt_finetuned"]
        - merged["logp_alt_pretrained"]
    )

    merged["delta_logp_wt_finetuned_minus_pretrained"] = (
        merged["logp_wt_finetuned"]
        - merged["logp_wt_pretrained"]
    )

    merged["delta_p_alt_finetuned_minus_pretrained"] = (
        merged["p_alt_finetuned"]
        - merged["p_alt_pretrained"]
    )

    merged["delta_p_wt_finetuned_minus_pretrained"] = (
        merged["p_wt_finetuned"]
        - merged["p_wt_pretrained"]
    )

    merged["alt_rank_improved"] = (
        merged["alt_rank_finetuned"]
        < merged["alt_rank_pretrained"]
    )

    merged["wt_rank_worsened"] = (
        merged["wt_rank_finetuned"]
        > merged["wt_rank_pretrained"]
    )

    if args.comparison_label is not None:
        merged["comparison_label"] = args.comparison_label

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    full_out = Path(str(out_prefix) + "_full.csv")
    summary_out = Path(str(out_prefix) + "_summary.csv")
    compact_out = Path(str(out_prefix) + "_compact.csv")

    merged.to_csv(full_out, index=False)

    compact_cols = [
        "dataset",
        "gene",
        "accession",
        "mutation",
        "wt",
        "alt",
        "reference_pos_1based",
        "homolog_mapped_pos_1based",
        "llr_alt_minus_wt_pretrained",
        "llr_alt_minus_wt_finetuned",
        "delta_llr_finetuned_minus_pretrained",
        "p_wt_pretrained",
        "p_wt_finetuned",
        "p_alt_pretrained",
        "p_alt_finetuned",
        "delta_p_alt_finetuned_minus_pretrained",
        "wt_rank_pretrained",
        "wt_rank_finetuned",
        "alt_rank_pretrained",
        "alt_rank_finetuned",
        "top_aa_pretrained",
        "top_aa_finetuned",
    ]

    compact_cols = [c for c in compact_cols if c in merged.columns]
    merged[compact_cols].to_csv(compact_out, index=False)

    summary = (
        merged.groupby(["dataset", "gene", "mutation"])
        .agg(
            n=("delta_llr_finetuned_minus_pretrained", "size"),
            mean_pretrained_llr=("llr_alt_minus_wt_pretrained", "mean"),
            median_pretrained_llr=("llr_alt_minus_wt_pretrained", "median"),
            std_pretrained_llr=("llr_alt_minus_wt_pretrained", "std"),
            mean_finetuned_llr=("llr_alt_minus_wt_finetuned", "mean"),
            median_finetuned_llr=("llr_alt_minus_wt_finetuned", "median"),
            std_finetuned_llr=("llr_alt_minus_wt_finetuned", "std"),
            mean_delta_llr=("delta_llr_finetuned_minus_pretrained", "mean"),
            median_delta_llr=("delta_llr_finetuned_minus_pretrained", "median"),
            std_delta_llr=("delta_llr_finetuned_minus_pretrained", "std"),
            mean_pretrained_p_wt=("p_wt_pretrained", "mean"),
            mean_finetuned_p_wt=("p_wt_finetuned", "mean"),
            mean_pretrained_p_alt=("p_alt_pretrained", "mean"),
            mean_finetuned_p_alt=("p_alt_finetuned", "mean"),
            mean_delta_p_alt=("delta_p_alt_finetuned_minus_pretrained", "mean"),
            frac_delta_llr_positive=("delta_llr_finetuned_minus_pretrained", lambda x: (x > 0).mean()),
            frac_alt_rank_improved=("alt_rank_improved", "mean"),
        )
        .reset_index()
    )

    if args.comparison_label is not None:
        summary.insert(0, "comparison_label", args.comparison_label)

    summary.to_csv(summary_out, index=False)

    print()
    print("Summary:")
    print(summary)

    print()
    print(f"Wrote full comparison:    {full_out}")
    print(f"Wrote compact comparison: {compact_out}")
    print(f"Wrote summary:            {summary_out}")


if __name__ == "__main__":
    main()
