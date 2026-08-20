#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/mmseqs40 tmp/mmseqs40

THREADS="${THREADS:-16}"

for gene in topA yeiB spoT dgoA; do
    echo "==== MMseqs2 40% clustering for ${gene} ===="

    input="data/processed/deduped_blast_derived_homologs/${gene}_deduped.faa"

    db="results/mmseqs40/${gene}_DB"
    clu="results/mmseqs40/${gene}_40_clu"
    repdb="results/mmseqs40/${gene}_40_repDB"
    tmpdir="tmp/mmseqs40/${gene}"

    rm -rf "$db"* "$clu"* "$repdb"* "$tmpdir"
    mkdir -p "$tmpdir"

    mmseqs createdb "$input" "$db"

    mmseqs cluster "$db" "$clu" "$tmpdir" \
        --min-seq-id 0.40 \
        -c 0.80 \
        --cov-mode 0 \
        --threads "$THREADS"

    mmseqs createtsv "$db" "$db" "$clu" \
        "results/mmseqs40/${gene}_40_clusters.tsv"

    mmseqs createsubdb "$clu" "$db" "$repdb"

    mmseqs convert2fasta "$repdb" \
        "results/mmseqs40/${gene}_40_representatives.faa"

    echo "Done: ${gene}"
done
