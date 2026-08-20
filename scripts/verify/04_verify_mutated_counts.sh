#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell
mkdir -p reports/verify

out="reports/verify/cluster95_mutated_counts.tsv"

echo -e "gene\tsplit\tmutated_sequence_count" > "$out"

for gene in topA yeiB spoT dgoA; do
    for split in train val test; do
        fasta="data/splits_cluster95/${gene}/mutated/${split}_mutated.faa"
        count=$(grep -c '^>' "$fasta")
        echo -e "${gene}\t${split}\t${count}" >> "$out"
    done
done

cat "$out"
echo
echo "Wrote: $out"
