#!/usr/bin/env bash
set -euo pipefail

echo "==== ESM-2 E. coli cluster95 analysis ===="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo

PROJECT_ROOT="/project/massrl/silbadowell"
cd "${PROJECT_ROOT}"

BASE_MODEL="facebook/esm2_t30_150M_UR50D"
FINETUNED_MODEL="pLM-ALE/runs/l40/esm2_150m_ecoli_cluster95_l40"

MAPPING_TSV="results/ale_site_scores/test_homolog_mapped_sites.tsv"
ECOLI_MAPPING_TSV="results/ale_site_scores/test_homolog_mapped_sites_ecoli.tsv"

PRETRAINED_OUT="results/ale_site_scores/pretrained_esm2_150m_ecoli_test_homolog_sites.csv"
FINETUNED_OUT="results/ale_site_scores/finetuned_esm2_150m_ecoli_cluster95_l40_test_homolog_sites.csv"

COMPARE_PREFIX="results/ale_site_scores/esm2_150m_ecoli_pretrained_vs_finetuned_test_homolog_sites"
FIG_PREFIX="results/figures/ale_site_scores/esm2_150m_ecoli"

BATCH_SIZE="${BATCH_SIZE:-8}"

echo "Base model: ${BASE_MODEL}"
echo "Fine-tuned model: ${FINETUNED_MODEL}"
echo "Mapping TSV: ${MAPPING_TSV}"
echo "Batch size: ${BATCH_SIZE}"
echo

mkdir -p results/ale_site_scores
mkdir -p results/figures/ale_site_scores
mkdir -p results/figures/window_llr_heatmaps

echo "==== Creating E. coli-only mapping TSV ===="
python - <<PY2
import pandas as pd
src = "${MAPPING_TSV}"
dst = "${ECOLI_MAPPING_TSV}"
df = pd.read_csv(src, sep="\\t")
df = df[df["dataset"] == "ecoli"].copy()
df.to_csv(dst, sep="\\t", index=False)
print(f"Wrote {dst} with {len(df)} rows")
print(df.groupby(["gene", "mutation"]).size().to_string())
PY2
echo

echo "==== Checking required files ===="

for f in \
    "${MAPPING_TSV}" \
    "${ECOLI_MAPPING_TSV}" \
    scripts/eval/score_test_homolog_sites_mlm.py \
    scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    scripts/plotting/plot_ale_site_llr_figures.py \
    scripts/plotting/plot_llr_heatmap_from_site_scores.py \
    scripts/plotting/plot_window_llr_heatmap_model_pair.py
do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: Missing required file/script: $f"
        exit 1
    fi
done

if [[ ! -d "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: Missing fine-tuned model directory: ${FINETUNED_MODEL}"
    exit 1
fi

if [[ ! -f "${FINETUNED_MODEL}/adapter_config.json" ]]; then
    echo "ERROR: Missing adapter_config.json in ${FINETUNED_MODEL}"
    exit 1
fi

echo "Checks passed."
echo

echo "==== 1. Score pretrained ESM-2 on E. coli held-out homolog sites ===="
python scripts/eval/score_test_homolog_sites_mlm.py \
    --model "${BASE_MODEL}" \
    --model_label pretrained_esm2_150m \
    --mapping_tsv "${ECOLI_MAPPING_TSV}" \
    --out "${PRETRAINED_OUT}" \
    --batch_size "${BATCH_SIZE}"

echo

echo "==== 2. Score fine-tuned ESM-2 on E. coli held-out homolog sites ===="
python scripts/eval/score_test_homolog_sites_mlm.py \
    --model "${FINETUNED_MODEL}" \
    --base_model "${BASE_MODEL}" \
    --model_label finetuned_esm2_150m_ecoli_cluster95_l40 \
    --mapping_tsv "${ECOLI_MAPPING_TSV}" \
    --out "${FINETUNED_OUT}" \
    --batch_size "${BATCH_SIZE}"

echo

echo "==== 3. Compare pretrained vs fine-tuned ESM-2 ===="
python scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    --pretrained_csv "${PRETRAINED_OUT}" \
    --finetuned_csv "${FINETUNED_OUT}" \
    --out_prefix "${COMPARE_PREFIX}" \
    --comparison_label esm2_150m_ecoli_cluster95_l40

echo

echo "==== 4. Print summary ===="
python - <<PY
import pandas as pd

summary = "${COMPARE_PREFIX}_summary.csv"
df = pd.read_csv(summary)

cols = [
    "mutation",
    "n",
    "mean_pretrained_llr",
    "mean_finetuned_llr",
    "mean_delta_llr",
    "median_delta_llr",
    "mean_pretrained_p_alt",
    "mean_finetuned_p_alt",
    "mean_delta_p_alt",
    "frac_delta_llr_positive",
    "frac_alt_rank_improved",
]

cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
PY

echo

echo "==== 5. Generate main/supporting/supplementary LLR figures ===="
python scripts/plotting/plot_ale_site_llr_figures.py \
    --summary_csv "${COMPARE_PREFIX}_summary.csv" \
    --full_csv "${COMPARE_PREFIX}_full.csv" \
    --out_prefix "${FIG_PREFIX}" \
    --title_prefix "ESM-2 E. coli"

echo

echo "==== 6. Generate amino-acid LLR heatmaps from held-out homolog scores ===="
python scripts/plotting/plot_llr_heatmap_from_site_scores.py \
    --comparison_full_csv "${COMPARE_PREFIX}_full.csv" \
    --out_prefix "${FIG_PREFIX}" \
    --dataset_filter ecoli

echo

echo "==== 7. Generate local window LLR heatmaps ===="
for item in "topA H33Y" "yeiB L143I" "spoT K662I"; do
    gene=$(echo "$item" | awk '{print $1}')
    mut=$(echo "$item" | awk '{print $2}')

    echo "---- ${gene} ${mut} ----"

    python scripts/plotting/plot_window_llr_heatmap_model_pair.py \
        --model_type mlm \
        --pretrained_model "${BASE_MODEL}" \
        --finetuned_model "${FINETUNED_MODEL}" \
        --finetuned_base_model "${BASE_MODEL}" \
        --pretrained_label "ESM-2 pretrained" \
        --finetuned_label "ESM-2 fine-tuned" \
        --gene "${gene}" \
        --mutation "${mut}" \
        --window 10 \
        --out_prefix "results/figures/window_llr_heatmaps/esm2_ecoli_${gene}_${mut}_window10_clean"
done

echo

echo "==== Final output check ===="
ls -lh \
    "${PRETRAINED_OUT}" \
    "${FINETUNED_OUT}" \
    "${COMPARE_PREFIX}_full.csv" \
    "${COMPARE_PREFIX}_compact.csv" \
    "${COMPARE_PREFIX}_summary.csv"

echo
echo "Figures:"
ls -lh "${FIG_PREFIX}"*.png "${FIG_PREFIX}"*.pdf 2>/dev/null || true

echo
echo "Window heatmaps:"
ls -lh results/figures/window_llr_heatmaps/esm2_ecoli_*window10_clean*.png \
       results/figures/window_llr_heatmaps/esm2_ecoli_*window10_clean*.pdf 2>/dev/null || true

echo
echo "==== Done ===="
echo "End time: $(date)"
