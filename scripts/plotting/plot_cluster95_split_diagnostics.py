#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GENES_DEFAULT = ["topA", "yeiB", "spoT", "dgoA"]
SPLIT_ORDER = ["train", "val", "test"]


def find_col(columns, candidates, contains=None):
    cols = list(columns)

    for c in candidates:
        if c in cols:
            return c

    if contains:
        for c in cols:
            lc = str(c).lower()
            if all(x in lc for x in contains):
                return c

    return None


def read_split_table(path):
    df = pd.read_csv(path, sep="\t")

    split_col = find_col(df.columns, ["split"], contains=["split"])
    cluster_col = find_col(
        df.columns,
        ["cluster", "cluster_id", "cluster95", "cluster_rep", "representative"],
        contains=["cluster"],
    )

    if split_col is None:
        raise ValueError(f"Could not find split column in {path}. Columns: {list(df.columns)}")

    if cluster_col is None:
        raise ValueError(f"Could not find cluster column in {path}. Columns: {list(df.columns)}")

    df = df.copy()
    df["split"] = df[split_col].astype(str)
    df["cluster_id"] = df[cluster_col].astype(str)

    return df


def read_mmseqs_cluster_file(path):
    """
    Reads a standard MMseqs cluster TSV:
      representative<TAB>member
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#")

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    df = df.iloc[:, :2].copy()
    df.columns = ["cluster_id", "member_id"]
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["member_id"] = df["member_id"].astype(str)

    return df


def identity_from_filename(path):
    name = Path(path).name

    # Handles names like topA_95_clusters.tsv, topA_cluster95_clusters.tsv, etc.
    for ident in [40, 70, 80, 90, 95]:
        if re.search(rf"(^|[_-])(?:cluster)?{ident}($|[_%-])", name):
            return ident

    m = re.search(r"(\d{2})", name)
    if m:
        val = int(m.group(1))
        if val in [40, 70, 80, 90, 95]:
            return val

    return None


def summarize_split_tables(split_dir, genes):
    sequence_rows = []
    cluster_rows = []
    top_cluster_rows = []
    leakage_rows = []

    for gene in genes:
        path = Path(split_dir) / f"{gene}_cluster95_split.tsv"
        if not path.exists():
            print(f"Skipping missing split table: {path}")
            continue

        df = read_split_table(path)

        # Sequence counts by split.
        split_counts = df["split"].value_counts().to_dict()
        row = {"gene": gene}
        for split in SPLIT_ORDER:
            row[split] = int(split_counts.get(split, 0))
        row["total"] = int(len(df))
        sequence_rows.append(row)

        # Cluster assignment check.
        cluster_to_n_splits = df.groupby("cluster_id")["split"].nunique()
        bad_clusters = cluster_to_n_splits[cluster_to_n_splits > 1]

        leakage_rows.append({
            "gene": gene,
            "clusters_total": int(df["cluster_id"].nunique()),
            "clusters_spanning_multiple_splits": int(len(bad_clusters)),
            "sequence_rows": int(len(df)),
        })

        # Cluster counts by split.
        cluster_split = (
            df[["cluster_id", "split"]]
            .drop_duplicates()
            .groupby("split")
            .size()
            .to_dict()
        )

        crow = {"gene": gene}
        for split in SPLIT_ORDER:
            crow[split] = int(cluster_split.get(split, 0))
        crow["total"] = int(df["cluster_id"].nunique())
        cluster_rows.append(crow)

        # Top clusters.
        cluster_sizes = (
            df.groupby(["cluster_id", "split"])
            .size()
            .reset_index(name="size")
            .sort_values("size", ascending=False)
        )

        cluster_sizes.insert(0, "gene", gene)
        top_cluster_rows.append(cluster_sizes.head(25))

    seq_df = pd.DataFrame(sequence_rows)
    clust_df = pd.DataFrame(cluster_rows)
    leakage_df = pd.DataFrame(leakage_rows)

    if top_cluster_rows:
        top_df = pd.concat(top_cluster_rows, ignore_index=True)
    else:
        top_df = pd.DataFrame(columns=["gene", "cluster_id", "split", "size"])

    return seq_df, clust_df, top_df, leakage_df


def summarize_mmseqs_sweep(sweep_dir, genes):
    rows = []

    for gene in genes:
        files = sorted(Path(sweep_dir).glob(f"{gene}*clusters.tsv"))

        for path in files:
            identity = identity_from_filename(path)
            if identity is None:
                print(f"Could not infer identity threshold from filename, skipping: {path}")
                continue

            df = read_mmseqs_cluster_file(path)
            cluster_sizes = df.groupby("cluster_id")["member_id"].nunique()

            n_sequences = int(df["member_id"].nunique())
            n_clusters = int(cluster_sizes.shape[0])
            largest = int(cluster_sizes.max())
            singletons = int((cluster_sizes == 1).sum())

            rows.append({
                "gene": gene,
                "identity_threshold": identity,
                "sequences": n_sequences,
                "clusters": n_clusters,
                "largest_cluster": largest,
                "largest_cluster_fraction": largest / n_sequences if n_sequences else np.nan,
                "singleton_clusters": singletons,
            })

    if rows:
        return pd.DataFrame(rows).sort_values(["gene", "identity_threshold"]).reset_index(drop=True)

    return pd.DataFrame(columns=[
        "gene",
        "identity_threshold",
        "sequences",
        "clusters",
        "largest_cluster",
        "largest_cluster_fraction",
        "singleton_clusters",
    ])


def savefig(out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")

    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")


def plot_split_sequence_counts(seq_df, outdir):
    df = seq_df.copy()
    if df.empty:
        return

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    plt.figure(figsize=(7.2, 4.4))

    for split in SPLIT_ORDER:
        vals = df[split].values
        plt.bar(x, vals, bottom=bottom, label=split)
        bottom += vals

    plt.xticks(x, df["gene"].tolist())
    plt.ylabel("Sequences")
    plt.xlabel("Gene")
    plt.title("Cluster95 sequence counts by split")
    plt.legend(frameon=False)

    savefig(Path(outdir) / "cluster95_sequence_counts_by_split")


def plot_split_cluster_counts(clust_df, outdir):
    df = clust_df.copy()
    if df.empty:
        return

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    plt.figure(figsize=(7.2, 4.4))

    for split in SPLIT_ORDER:
        vals = df[split].values
        plt.bar(x, vals, bottom=bottom, label=split)
        bottom += vals

    plt.xticks(x, df["gene"].tolist())
    plt.ylabel("95% identity clusters")
    plt.xlabel("Gene")
    plt.title("Cluster95 cluster counts by split")
    plt.legend(frameon=False)

    savefig(Path(outdir) / "cluster95_cluster_counts_by_split")


def plot_top_clusters(top_df, outdir, top_n=20):
    if top_df.empty:
        return

    for gene, sub in top_df.groupby("gene"):
        sub = sub.sort_values("size", ascending=False).head(top_n).copy()
        sub = sub.iloc[::-1].copy()

        labels = [f"{cid[:10]}..." if len(str(cid)) > 10 else str(cid) for cid in sub["cluster_id"]]

        plt.figure(figsize=(7.5, max(4.0, 0.22 * len(sub) + 1.5)))
        plt.barh(np.arange(len(sub)), sub["size"].values)

        plt.yticks(np.arange(len(sub)), labels)
        plt.xlabel("Sequences in cluster")
        plt.ylabel("Cluster ID")
        plt.title(f"{gene}: largest 95% identity clusters")

        for i, (_, row) in enumerate(sub.iterrows()):
            plt.text(
                row["size"],
                i,
                f" {row['split']}",
                va="center",
                fontsize=8,
            )

        savefig(Path(outdir) / f"cluster95_top_clusters_{gene}")


def plot_cluster_size_distribution(split_dir, genes, outdir):
    plt.figure(figsize=(7.4, 4.6))

    any_data = False

    for gene in genes:
        path = Path(split_dir) / f"{gene}_cluster95_split.tsv"
        if not path.exists():
            continue

        df = read_split_table(path)
        sizes = df.groupby("cluster_id").size().values

        if len(sizes) == 0:
            continue

        any_data = True
        sorted_sizes = np.sort(sizes)[::-1]
        ranks = np.arange(1, len(sorted_sizes) + 1)

        plt.plot(ranks, sorted_sizes, marker="o", linewidth=1, markersize=2, label=gene)

    if not any_data:
        plt.close()
        return

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cluster rank")
    plt.ylabel("Sequences per cluster")
    plt.title("Cluster95 cluster-size distributions")
    plt.legend(frameon=False)

    savefig(Path(outdir) / "cluster95_cluster_size_rank_distribution")


def plot_mmseqs_sweep(sweep_df, outdir):
    if sweep_df.empty:
        print("No MMseqs sweep data found; skipping sweep plots.")
        return

    plt.figure(figsize=(7.4, 4.6))

    for gene, sub in sweep_df.groupby("gene"):
        sub = sub.sort_values("identity_threshold")
        plt.plot(
            sub["identity_threshold"],
            sub["clusters"],
            marker="o",
            linewidth=1.8,
            label=gene,
        )

    plt.xlabel("MMseqs2 identity threshold (%)")
    plt.ylabel("Number of clusters")
    plt.title("Homolog clustering across identity thresholds")
    plt.legend(frameon=False)

    savefig(Path(outdir) / "mmseqs_cluster_count_sweep")

    plt.figure(figsize=(7.4, 4.6))

    for gene, sub in sweep_df.groupby("gene"):
        sub = sub.sort_values("identity_threshold")
        plt.plot(
            sub["identity_threshold"],
            100.0 * sub["largest_cluster_fraction"],
            marker="o",
            linewidth=1.8,
            label=gene,
        )

    plt.xlabel("MMseqs2 identity threshold (%)")
    plt.ylabel("Largest cluster (% of sequences)")
    plt.title("Dominance of largest homolog cluster")
    plt.legend(frameon=False)

    savefig(Path(outdir) / "mmseqs_largest_cluster_fraction_sweep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="data/splits_cluster95/split_tables")
    parser.add_argument("--sweep_dir", default="results/mmseqs_sweep")
    parser.add_argument("--outdir", default="results/figures/homology_split")
    parser.add_argument("--genes", nargs="+", default=GENES_DEFAULT)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seq_df, clust_df, top_df, leakage_df = summarize_split_tables(
        split_dir=args.split_dir,
        genes=args.genes,
    )

    sweep_df = summarize_mmseqs_sweep(
        sweep_dir=args.sweep_dir,
        genes=args.genes,
    )

    seq_csv = outdir / "cluster95_sequence_counts_by_split.csv"
    clust_csv = outdir / "cluster95_cluster_counts_by_split.csv"
    top_csv = outdir / "cluster95_top_clusters.csv"
    leakage_csv = outdir / "cluster95_split_integrity_check.csv"
    sweep_csv = outdir / "mmseqs_cluster_sweep_summary.csv"

    seq_df.to_csv(seq_csv, index=False)
    clust_df.to_csv(clust_csv, index=False)
    top_df.to_csv(top_csv, index=False)
    leakage_df.to_csv(leakage_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)

    print(f"Wrote: {seq_csv}")
    print(f"Wrote: {clust_csv}")
    print(f"Wrote: {top_csv}")
    print(f"Wrote: {leakage_csv}")
    print(f"Wrote: {sweep_csv}")
    print()

    print("Cluster95 split integrity:")
    if not leakage_df.empty:
        print(leakage_df.to_string(index=False))
    else:
        print("No split integrity rows found.")
    print()

    plot_split_sequence_counts(seq_df, outdir)
    plot_split_cluster_counts(clust_df, outdir)
    plot_top_clusters(top_df, outdir)
    plot_cluster_size_distribution(args.split_dir, args.genes, outdir)
    plot_mmseqs_sweep(sweep_df, outdir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
