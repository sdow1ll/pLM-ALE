#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

bash scripts/verify/01_verify_intermediate_counts.sh
bash scripts/verify/02_verify_cluster_sweep.sh
bash scripts/verify/03_verify_cluster95_split_counts.sh
bash scripts/verify/04_verify_mutated_counts.sh
bash scripts/verify/05_verify_final_dataset_counts.sh

echo
echo "All light verification checks completed."
echo "Reports written to: reports/verify/"
