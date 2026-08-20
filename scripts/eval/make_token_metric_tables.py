#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, ndigits=3):
    return f"{float(x):.{ndigits}f}"


def make_prediction_table(metrics):
    rows = []

    for model_name, pre, ft in metrics:
        rows.append({
            "Model": model_name,
            "State": "Pretrained",
            "Accuracy": pre["accuracy"],
            "Recall": pre["macro_recall"],
            "Precision": pre["macro_precision"],
            "F1 Score": pre["macro_f1"],
        })
        rows.append({
            "Model": model_name,
            "State": "Fine-tuned",
            "Accuracy": ft["accuracy"],
            "Recall": ft["macro_recall"],
            "Precision": ft["macro_precision"],
            "F1 Score": ft["macro_f1"],
        })

    return pd.DataFrame(rows)


def make_ppl_table(metrics, ppl_key):
    rows = []

    row_pre = {"State": "Pretrained"}
    row_ft = {"State": "Fine-tuned"}
    row_delta = {"State": "$\\Delta$"}
    row_pct = {"State": "\\% Change"}

    for model_name, pre, ft in metrics:
        pre_ppl = float(pre[ppl_key])
        ft_ppl = float(ft[ppl_key])
        delta = ft_ppl - pre_ppl
        pct = 100.0 * delta / pre_ppl

        row_pre[model_name] = pre_ppl
        row_ft[model_name] = ft_ppl
        row_delta[model_name] = delta
        row_pct[model_name] = pct

    rows.extend([row_pre, row_ft, row_delta, row_pct])
    return pd.DataFrame(rows)


def write_prediction_latex(df, out_path):
    models = list(df["Model"].unique())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Pretrained and fine-tuned token prediction metrics on the cluster-controlled E. coli test set. ESM-2 was evaluated by masked-token recovery, while ProGen2 was evaluated by next-token prediction.}")
    lines.append("\\label{tab:ecoli_token_prediction_metrics_cluster95}")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\hline")

    for model in models:
        sub = df[df["Model"] == model].copy()

        lines.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{model}}}}} \\\\")
        lines.append("\\hline")
        lines.append("\\textbf{State} & \\textbf{Metric} & \\textbf{Value} & & \\\\")
        lines.append("\\hline")

        for _, row in sub.iterrows():
            state = row["State"]
            lines.append(f"\\textbf{{{state}}} & Accuracy  & {fmt(row['Accuracy'])} & & \\\\")
            lines.append(f" & Recall    & {fmt(row['Recall'])} & & \\\\")
            lines.append(f" & Precision & {fmt(row['Precision'])} & & \\\\")
            lines.append(f" & F1 Score  & {fmt(row['F1 Score'])} & & \\\\")
            lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def write_prediction_latex_like_original(df, out_path):
    models = list(df["Model"].unique())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Pretrained/fine-tuned masked-token and next-token prediction metrics for ESM-2 and ProGen2 on the cluster-controlled E. coli test set.}")
    lines.append("\\label{tab:ecoli_token_prediction_metrics_cluster95}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\hline")

    for model in models:
        sub = df[df["Model"] == model].copy()

        lines.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{model}}}}} \\\\")
        lines.append("\\hline")
        lines.append("\\textbf{Metric} & \\textbf{Pretrained} & \\textbf{Fine-tuned} \\\\")
        lines.append("\\hline")

        pre = sub[sub["State"] == "Pretrained"].iloc[0]
        ft = sub[sub["State"] == "Fine-tuned"].iloc[0]

        for metric in ["Accuracy", "Recall", "Precision", "F1 Score"]:
            lines.append(
                f"{metric} & {fmt(pre[metric])} & \\textbf{{{fmt(ft[metric])}}} \\\\"
            )

        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def write_ppl_latex(df, out_path):
    model_cols = [c for c in df.columns if c != "State"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Average pseudo-perplexity for ESM-2 and ProGen2 before and after fine-tuning on the cluster-controlled E. coli test set.}")
    lines.append("\\label{tab:ecoli_pseudoperplexity_cluster95}")
    lines.append("\\begin{tabular}{l" + "r" * len(model_cols) + "}")
    lines.append("\\hline")
    lines.append(" & " + " & ".join([f"\\textbf{{{m}}}" for m in model_cols]) + " \\\\")
    lines.append("\\hline")

    for _, row in df.iterrows():
        state = row["State"]
        vals = []
        for m in model_cols:
            vals.append(fmt(row[m]))
        lines.append(f"\\textbf{{{state}}} & " + " & ".join(vals) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2_pretrained_json", required=True)
    parser.add_argument("--esm2_finetuned_json", required=True)
    parser.add_argument("--progen2_pretrained_json", required=True)
    parser.add_argument("--progen2_finetuned_json", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument(
        "--ppl_key",
        default="mean_sequence_pseudo_perplexity",
        choices=["mean_sequence_pseudo_perplexity", "corpus_pseudo_perplexity"],
    )
    args = parser.parse_args()

    metrics = [
        (
            "ESM-2",
            load_json(args.esm2_pretrained_json),
            load_json(args.esm2_finetuned_json),
        ),
        (
            "ProGen2",
            load_json(args.progen2_pretrained_json),
            load_json(args.progen2_finetuned_json),
        ),
    ]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred_df = make_prediction_table(metrics)
    ppl_df = make_ppl_table(metrics, args.ppl_key)

    pred_csv = out_prefix.with_name(out_prefix.name + "_token_prediction_metrics.csv")
    ppl_csv = out_prefix.with_name(out_prefix.name + "_pseudoperplexity.csv")

    pred_latex = out_prefix.with_name(out_prefix.name + "_token_prediction_metrics.tex")
    pred_latex_original = out_prefix.with_name(out_prefix.name + "_token_prediction_metrics_like_original.tex")
    ppl_latex = out_prefix.with_name(out_prefix.name + "_pseudoperplexity.tex")

    pred_df.to_csv(pred_csv, index=False)
    ppl_df.to_csv(ppl_csv, index=False)

    write_prediction_latex(pred_df, pred_latex)
    write_prediction_latex_like_original(pred_df, pred_latex_original)
    write_ppl_latex(ppl_df, ppl_latex)

    print("Prediction metrics:")
    print(pred_df.to_string(index=False))
    print()
    print("Pseudo-perplexity:")
    print(ppl_df.to_string(index=False))
    print()
    print(f"Wrote: {pred_csv}")
    print(f"Wrote: {ppl_csv}")
    print(f"Wrote: {pred_latex}")
    print(f"Wrote: {pred_latex_original}")
    print(f"Wrote: {ppl_latex}")


if __name__ == "__main__":
    main()
