#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

BASE_MODEL="hugohrban/progen2-small"

MUT="${1:-F33I}"

MODEL_DIR="pLM-ALE/runs/mutation_holdout/progen2_151m_dgoa_holdout_${MUT}_l40"

# If the run saved only checkpoints, use latest checkpoint.
if [[ -f "${MODEL_DIR}/adapter_config.json" ]]; then
    FT_MODEL="${MODEL_DIR}"
else
    FT_MODEL="$(find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi

MAPPING_TSV="data/mutation_holdout/dgoA/holdout_${MUT}/test_heldout_mapping.tsv"

PRETRAINED_OUT="results/ale_site_scores/mutation_holdout_pretrained_progen2_151m_dgoa_holdout_${MUT}.csv"
FINETUNED_OUT="results/ale_site_scores/mutation_holdout_finetuned_progen2_151m_dgoa_holdout_${MUT}.csv"
COMPARE_PREFIX="results/ale_site_scores/mutation_holdout_progen2_151m_dgoa_holdout_${MUT}"

echo "==== DgoA mutation-held-out evaluation ===="
echo "Held-out mutation: ${MUT}"
echo "Base model: ${BASE_MODEL}"
echo "Fine-tuned model: ${FT_MODEL}"
echo "Mapping TSV: ${MAPPING_TSV}"
echo

if [[ ! -f "${MAPPING_TSV}" ]]; then
    echo "ERROR: Missing mapping TSV: ${MAPPING_TSV}"
    exit 1
fi

if [[ ! -d "${FT_MODEL}" ]]; then
    echo "ERROR: Missing fine-tuned model/checkpoint directory: ${FT_MODEL}"
    exit 1
fi

if [[ ! -f "${FT_MODEL}/adapter_config.json" ]]; then
    echo "ERROR: Missing adapter_config.json in ${FT_MODEL}"
    exit 1
fi

echo "==== Mapping rows ===="
python - <<PY
import pandas as pd
f = "${MAPPING_TSV}"
df = pd.read_csv(f, sep="\t")
print(df.groupby(["gene", "mutation"]).size().to_string())
print("rows:", len(df))
PY
echo

echo "==== 1. Score pretrained ProGen2 on held-out ${MUT} ===="
python scripts/eval/score_test_homolog_sites_causal.py \
    --model "${BASE_MODEL}" \
    --model_label "pretrained_progen2_151m_dgoa_holdout_${MUT}" \
    --mapping_tsv "${MAPPING_TSV}" \
    --out "${PRETRAINED_OUT}" \
    --batch_size 4

echo

echo "==== 2. Score fine-tuned holdout model on held-out ${MUT} ===="
python scripts/eval/score_test_homolog_sites_causal.py \
    --model "${FT_MODEL}" \
    --base_model "${BASE_MODEL}" \
    --model_label "finetuned_progen2_151m_dgoa_holdout_${MUT}" \
    --mapping_tsv "${MAPPING_TSV}" \
    --out "${FINETUNED_OUT}" \
    --batch_size 4

echo

echo "==== 3. Compare pretrained vs fine-tuned ===="
python scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    --pretrained_csv "${PRETRAINED_OUT}" \
    --finetuned_csv "${FINETUNED_OUT}" \
    --out_prefix "${COMPARE_PREFIX}" \
    --comparison_label "mutation_holdout_progen2_151m_dgoa_holdout_${MUT}"

echo

echo "==== Summary ===="
python - <<PY
import pandas as pd

f = "${COMPARE_PREFIX}_summary.csv"
df = pd.read_csv(f)

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
echo "==== Wrote outputs ===="
echo "${PRETRAINED_OUT}"
echo "${FINETUNED_OUT}"
echo "${COMPARE_PREFIX}_full.csv"
echo "${COMPARE_PREFIX}_compact.csv"
echo "${COMPARE_PREFIX}_summary.csv"
