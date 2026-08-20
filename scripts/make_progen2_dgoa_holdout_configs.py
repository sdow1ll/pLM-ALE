#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

MUTATIONS = ["F33I", "D58N", "Q72H", "A75V", "V85A", "V154F", "Y180F"]

BAD_TOP_LEVEL_KEYS = {
    "test_path",
    "heldout_mutation",
    "wandb_notes",
}

def make_yaml_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_yaml_safe(v) for v in obj]
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)

def clean_config(cfg):
    cfg = dict(cfg)

    for key in BAD_TOP_LEVEL_KEYS:
        cfg.pop(key, None)

    if "training_args" in cfg and isinstance(cfg["training_args"], dict):
        cfg["training_args"] = {
            k: v for k, v in cfg["training_args"].items()
            if not str(k).startswith("_")
        }

    return make_yaml_safe(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        required=True,
        help="Known-working ProGen2 DgoA config to copy and modify."
    )
    parser.add_argument(
        "--outdir",
        default="configs/mutation_holdout/progen2_dgoa",
    )
    parser.add_argument(
        "--data_root",
        default="../data/mutation_holdout/dgoA",
        help="Path from pLM-ALE repo root to mutation-holdout data."
    )
    parser.add_argument(
        "--output_root",
        default="runs/mutation_holdout",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    outdir = Path(args.outdir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if not template_path.exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    with open(template_path) as f:
        cfg_template = yaml.load(f, Loader=yaml.UnsafeLoader)

    if not isinstance(cfg_template, dict):
        raise ValueError("Template config did not load as a dictionary.")

    for mut in MUTATIONS:
        cfg = clean_config(cfg_template)

        holdout_dir = data_root / f"holdout_{mut}"
        train_path = holdout_dir / "train.faa"
        eval_path = holdout_dir / "val.faa"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing train FASTA for {mut}: {train_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing val FASTA for {mut}: {eval_path}")

        cfg["base_model"] = "hugohrban/progen2-small"
        cfg["train_path"] = str(train_path)
        cfg["eval_path"] = str(eval_path)

        if "wandb_project" in cfg:
            cfg["wandb_project"] = "pLM-ALE-mutation-holdout"
        if "wandb_run_name" in cfg:
            cfg["wandb_run_name"] = f"progen2_dgoa_holdout_{mut}"

        output_dir = output_root / f"progen2_151m_dgoa_holdout_{mut}_l40"

        if "training_args" not in cfg or not isinstance(cfg["training_args"], dict):
            cfg["training_args"] = {}

        cfg["training_args"]["output_dir"] = str(output_dir)
        cfg["training_args"]["run_name"] = f"progen2_dgoa_holdout_{mut}"

        # Requested training settings
        cfg["training_args"]["num_train_epochs"] = 20
        cfg["training_args"]["logging_steps"] = 100
        cfg["training_args"]["eval_steps"] = 100
        cfg["training_args"]["save_steps"] = 100
        cfg["training_args"]["evaluation_strategy"] = "steps"
        cfg["training_args"]["save_strategy"] = "steps"

        # Keep best checkpoint if your trainer supports it.
        cfg["training_args"]["load_best_model_at_end"] = True
        cfg["training_args"]["metric_for_best_model"] = "eval_loss"
        cfg["training_args"]["greater_is_better"] = False

        out = outdir / f"progen2_dgoa_holdout_{mut}.yml"

        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {out}")
        print(f"  held out:   {mut}")
        print(f"  train_path: {train_path}")
        print(f"  eval_path:  {eval_path}")
        print(f"  output_dir: {output_dir}")
        print(f"  epochs:     20")
        print(f"  steps:      100")

if __name__ == "__main__":
    main()
