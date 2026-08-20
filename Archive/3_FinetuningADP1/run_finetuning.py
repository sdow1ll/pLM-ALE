# run_finetuning.py
import yaml
import subprocess

subprocess.run(["python", "finetuneESM2_ProGen2_LoRA.py", "--config", "optimized-config.yml"])
