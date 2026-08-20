#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell
mkdir -p reports/verify

out="reports/verify/final_cluster95_dataset_counts.tsv"

echo -e "dataset\tsplit\tsequence_count" > "$out"

for dataset in ecoli dgoA; do
    for split in train val test; do
        fasta="data/final_cluster95/${dataset}/${split}.faa"
        count=$(grep -c '^>' "$fasta")
        echo -e "${dataset}\t${split}\t${count}" >> "$out"
    done
done

cat "$out"
echo
echo "Wrote: $out"
