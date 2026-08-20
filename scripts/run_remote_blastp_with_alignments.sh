#!/usr/bin/env bash
set -euo pipefail

mkdir -p blast_results

for gene in topA yeiB spoT dgoA; do
    echo "==== Running remote BLASTP with alignments for ${gene} ===="

    blastp \
        -query "data/raw/queries/by_gene/${gene}_query.faa" \
        -db refseq_protein \
        -remote \
        -max_target_seqs 10000 \
        -outfmt "6 sacc pident qcovs length qlen slen qstart qend sstart send evalue bitscore qseq sseq stitle" \
        -out "blast_results/${gene}_blast_align.tsv"

    echo "Done: blast_results/${gene}_blast_align.tsv"
done
