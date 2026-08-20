#!/usr/bin/env bash
set -euo pipefail

THREADS="${THREADS:-16}"

mkdir -p results/mmseqs_sweep tmp/mmseqs_sweep

for ident in 0.40 0.70 0.80 0.90 0.95; do
    tag=$(echo "$ident" | sed 's/0\.//')

    for gene in topA yeiB spoT dgoA; do
        echo "==== ${gene} clustering at ${ident} identity ===="

        input="data/processed/deduped_blast_derived_homologs/${gene}_deduped.faa"

        db="results/mmseqs_sweep/${gene}_${tag}_DB"
        clu="results/mmseqs_sweep/${gene}_${tag}_clu"
        tmpdir="tmp/mmseqs_sweep/${gene}_${tag}"

        rm -rf "$db"* "$clu"* "$tmpdir"
        mkdir -p "$tmpdir"

        mmseqs createdb "$input" "$db"

        mmseqs cluster "$db" "$clu" "$tmpdir" \
            --min-seq-id "$ident" \
            -c 0.80 \
            --cov-mode 0 \
            --threads "$THREADS"

        mmseqs createtsv "$db" "$db" "$clu" \
            "results/mmseqs_sweep/${gene}_${tag}_clusters.tsv"

        echo "Done: ${gene} ${ident}"
    done
done
