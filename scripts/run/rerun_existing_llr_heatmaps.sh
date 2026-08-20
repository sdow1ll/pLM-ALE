#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

mkdir -p results/figures/ale_site_scores

run_heatmap () {
    local dataset="$1"
    local full_csv="$2"
    local out_prefix="$3"

    echo
    echo "==== ${out_prefix} ===="
    echo "Dataset: ${dataset}"
    echo "Full CSV: ${full_csv}"

    if [[ ! -f "${full_csv}" ]]; then
        echo "Skipping; missing: ${full_csv}"
        return 0
    fi

    python scripts/plotting/plot_llr_heatmap_from_site_scores.py \
        --comparison_full_csv "${full_csv}" \
        --out_prefix "${out_prefix}" \
        --dataset_filter "${dataset}"
}

# Main same-dataset analyses.
run_heatmap \
    ecoli \
    results/ale_site_scores/progen2_151m_ecoli_pretrained_vs_finetuned_test_homolog_sites_full.csv \
    results/figures/ale_site_scores/progen2_151m_ecoli

run_heatmap \
    dgoA \
    results/ale_site_scores/progen2_151m_dgoa_pretrained_vs_finetuned_test_homolog_sites_full.csv \
    results/figures/ale_site_scores/progen2_151m_dgoa

run_heatmap \
    ecoli \
    results/ale_site_scores/esm2_150m_ecoli_pretrained_vs_finetuned_test_homolog_sites_full.csv \
    results/figures/ale_site_scores/esm2_150m_ecoli

run_heatmap \
    dgoA \
    results/ale_site_scores/esm2_150m_dgoa_pretrained_vs_finetuned_test_homolog_sites_full.csv \
    results/figures/ale_site_scores/esm2_150m_dgoa

# Transfer analyses, if those CSVs exist.
run_heatmap \
    dgoA \
    results/ale_site_scores/progen2_151m_ecoli_finetuned_transfer_to_dgoa_full.csv \
    results/figures/ale_site_scores/progen2_151m_ecoli_transfer_to_dgoa

run_heatmap \
    ecoli \
    results/ale_site_scores/progen2_151m_dgoa_finetuned_transfer_to_ecoli_full.csv \
    results/figures/ale_site_scores/progen2_151m_dgoa_transfer_to_ecoli

echo
echo "==== Finished rerunning existing heatmaps ===="
