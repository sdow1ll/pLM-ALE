#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell
mkdir -p reports/verify

python scripts/summarize_cluster_sweep.py \
    | tee reports/verify/cluster_sweep_summary.txt

cp results/mmseqs_sweep/cluster_sweep_summary.tsv \
   reports/verify/cluster_sweep_summary.tsv

echo
echo "Wrote:"
echo "  reports/verify/cluster_sweep_summary.txt"
echo "  reports/verify/cluster_sweep_summary.tsv"
