#!/usr/bin/env bash
set -euo pipefail

mkdir -p blast_results

for gene in topA yeiB spoT dgoA; do
    echo "==== Running remote BLASTP for ${gene} ===="

    blastp \
        -query "data/raw/queries/by_gene/${gene}_query.faa" \
        -db refseq_protein \
        -remote \
        -max_target_seqs 10000 \
        -outfmt "6 sacc pident qcovs length qlen slen evalue bitscore stitle" \
        -out "blast_results/${gene}_blast.tsv"

    echo "Done: blast_results/${gene}_blast.tsv"
done
