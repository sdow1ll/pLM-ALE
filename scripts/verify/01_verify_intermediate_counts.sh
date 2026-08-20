#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell
mkdir -p reports/verify

out="reports/verify/intermediate_counts.tsv"

echo -e "gene\tblast_derived_homologs\tdeduped_homologs\tmutation_map_rows" > "$out"

for gene in topA yeiB spoT dgoA; do
    blast_fasta="data/processed/blast_derived_homologs/${gene}_blast_derived_homologs.faa"
    dedup_fasta="data/processed/deduped_blast_derived_homologs/${gene}_deduped.faa"
    map_file="data/processed/blast_derived_mutation_maps/${gene}_mutation_map.tsv"

    blast_count=$(grep -c '^>' "$blast_fasta")
    dedup_count=$(grep -c '^>' "$dedup_fasta")
    map_rows=$(($(wc -l < "$map_file") - 1))

    echo -e "${gene}\t${blast_count}\t${dedup_count}\t${map_rows}" >> "$out"
done

cat "$out"
echo
echo "Wrote: $out"
