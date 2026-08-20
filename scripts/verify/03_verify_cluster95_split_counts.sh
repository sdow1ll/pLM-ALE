#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell
mkdir -p reports/verify

out="reports/verify/cluster95_split_counts.tsv"

echo -e "gene\tsplit\tsequence_count" > "$out"

for gene in topA yeiB spoT dgoA; do
    split_table="data/splits_cluster95/split_tables/${gene}_cluster95_split.tsv"

    awk -v gene="$gene" '
        BEGIN { FS=OFS="\t" }
        NR > 1 { counts[$3]++ }
        END {
            print gene, "train", counts["train"] + 0
            print gene, "val", counts["val"] + 0
            print gene, "test", counts["test"] + 0
        }
    ' "$split_table" >> "$out"
done

cat "$out"
echo
echo "Wrote: $out"
