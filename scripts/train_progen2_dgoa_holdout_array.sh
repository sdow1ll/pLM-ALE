#!/usr/bin/env bash
#SBATCH --job-name=progen2_dgoa_holdout
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --account=massrl
#SBATCH --partition=mb-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5

set -euo pipefail

echo "==== Job info ===="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo

# Activate conda.
# Temporarily disable nounset because /etc/bashrc may reference unset variables.
set +u
source ~/.bashrc
conda activate test
set -u

cd /project/massrl/silbadowell/pLM-ALE

# F33I is intentionally excluded.
MUTATIONS=(D58N Q72H A75V V85A V154F Y180F)
MUT="${MUTATIONS[$SLURM_ARRAY_TASK_ID]}"

CONFIG="configs/mutation_holdout/progen2_dgoa/progen2_dgoa_holdout_${MUT}.yml"

echo "==== Holdout fold ===="
echo "Held-out mutation: ${MUT}"
echo "Config: ${CONFIG}"
echo

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

OUTDIR="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}"))
print(cfg.get("training_args", {}).get("output_dir", ""))
PY
)"

echo "Output directory: ${OUTDIR}"
echo

if [[ -z "${OUTDIR}" ]]; then
    echo "ERROR: Could not read training_args.output_dir from ${CONFIG}"
    exit 1
fi

if [[ "${FORCE:-0}" != "1" ]]; then
    if [[ -f "${OUTDIR}/adapter_config.json" ]]; then
        echo "Model already appears complete: ${OUTDIR}"
        echo "Skipping. Set FORCE=1 to retrain."
        exit 0
    fi
fi

echo "==== Active config key fields ===="
grep -E "^(base_model|train_path|eval_path|wandb_project|wandb_run_name):" "${CONFIG}" || true
grep -A40 "^training_args:" "${CONFIG}" || true
echo

echo "==== Check config does not contain unsupported top-level keys ===="
if grep -E "^(test_path|heldout_mutation|wandb_notes):" "${CONFIG}"; then
    echo "ERROR: Config contains unsupported top-level keys."
    exit 1
fi
echo "Config key check passed."
echo

echo "==== Check requested training settings ===="
grep -E "num_train_epochs|logging_steps|eval_steps|save_steps|evaluation_strategy|save_strategy" "${CONFIG}" || true
echo

echo "==== Check FASTA inputs ===="
python - <<PY
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("${CONFIG}"))

for key in ["train_path", "eval_path"]:
    p = Path(cfg[key])
    print(f"{key}: {p}")
    print(f"  exists: {p.exists()}")
    if not p.exists():
        raise FileNotFoundError(p)
    n = sum(1 for line in open(p) if line.startswith(">"))
    print(f"  records: {n}")

print("FASTA input check passed.")
PY
echo

echo "==== GPU check ===="
nvidia-smi || true
echo

echo "==== Starting training ===="
python finetuneESM2_ProGen2_LoRA.py --config "${CONFIG}"

echo
echo "==== Finished ===="
echo "Held-out mutation: ${MUT}"
echo "Output directory: ${OUTDIR}"
echo "End time: $(date)"
