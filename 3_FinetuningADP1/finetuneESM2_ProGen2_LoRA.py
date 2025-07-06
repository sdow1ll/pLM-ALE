#!/usr/bin/env python
"""Finetune the ProGen and ESM model with LoRA for masked language modeling, including WandB logging."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Union

import torch
import transformers
import wandb
import yaml
from torch.utils.data import Dataset
from transformers import (
    EsmForMaskedLM,
    EsmTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint

# Import PEFT for LoRA integration
from peft import get_peft_model, LoraConfig, TaskType

# Setup basic logging for debugging and info messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Define a type for paths that can be either a string or a pathlib.Path
PathLike = Union[str, Path]


@dataclass
class Sequence:
    """Store a biological sequence and its description tag."""
    sequence: str
    tag: str

    def __hash__(self) -> int:
        # Hash based on sequence and tag for use in sets or dicts
        return hash((self.sequence, self.tag))


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """
    Read FASTA file sequences and description tags into a list of Sequence instances.
    Uses a regex to capture multi-line sequences.
    """
    fasta_text = Path(fasta_file).read_text()
    # Regex finds entries: everything after '>' until newline is the tag, followed by the sequence.
    entries = re.findall(r">(.*?)\n([^>]+)", fasta_text, re.DOTALL)
    sequences = []
    for tag, seq in entries:
        # Remove any whitespace/newlines from the sequence
        seq_clean = "".join(seq.split())
        sequences.append(Sequence(sequence=seq_clean, tag=tag.strip()))
    return sequences


def write_fasta(sequences: Union[Sequence, List[Sequence]], fasta_file: PathLike, mode: str = "w") -> None:
    """
    Write or append sequences to a FASTA file.
    """
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f">{seq.tag}\n{seq.sequence}\n")


def random_split_fasta(input_fasta: PathLike, output_dir: PathLike, split: float = 0.8, seed: int = 0) -> None:
    """
    Randomly split a FASTA file into training and validation FASTA files.
    Saves the split files and metadata about the split.
    """
    sequences = read_fasta(input_fasta)
    random.seed(seed)
    random.shuffle(sequences)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    split_idx = int(len(sequences) * split)
    write_fasta(sequences[:split_idx], output_dir / "train.fasta")
    write_fasta(sequences[split_idx:], output_dir / "valid.fasta")
    # Copy the original file for reference
    shutil.copy(input_fasta, output_dir / Path(input_fasta).name)

    # Save metadata about the split
    metadata = {
        "input_fasta": str(Path(input_fasta).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "seed": seed,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"FASTA split complete: {split_idx} train, {len(sequences)-split_idx} validation sequences.")


class SequenceDataset(Dataset):
    """Custom dataset for biological sequences."""
    def __init__(self, sequences: List[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        # Returns a single sequence as a string
        return self.sequences[idx]

'''
class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for masked language modeling using the ESM model.
    Extends the default DataCollatorForLanguageModeling with support for a custom training mode.
    """
    def __init__(self, train_mode: bool = False, **kwargs: Any) -> None:
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def __call__(self, examples: List[str]) -> Any:
        # Tokenize the batch of sequences
        batch = self.tokenizer(
            examples,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode and self.mlm,
        )
        # For training, apply MLM masking if enabled
        if self.train_mode and self.mlm:
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        # For evaluation/inference, simply copy the input_ids to labels, adjusting pad tokens
        elif not self.train_mode:
            batch["labels"] = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch
'''

from transformers import DataCollatorForLanguageModeling
from typing import Any, Dict, List, Optional, Union
import torch

class HybridDataCollator:
    """
    A flexible data collator that supports both masked language modeling (MLM) and 
    autoregressive (causal) language modeling for protein sequences.
    
    This allows switching between ESM-style bidirectional masked prediction and 
    ProGen2-style next-token prediction modes.
    """
    def __init__(
        self, 
        tokenizer,
        model_type: str = "mlm",    # Options: "mlm", "clm", or "auto"
        mlm_probability: float = 0.15,
        max_length: int = 1024,
        pad_to_multiple_of: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # For MLM masking operations
        if model_type == "mlm" or model_type == "auto":
            self.mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
                pad_to_multiple_of=pad_to_multiple_of
            )
        
    def __call__(self, examples: List[str]) -> Dict[str, torch.Tensor]:
        """Process a batch of protein sequences based on the selected model type."""
        # Tokenize the sequences
        batch = self.tokenizer(
            examples,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length" if self.max_length else "longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Return special tokens mask only if needed for MLM
            return_special_tokens_mask=self.model_type in ["mlm", "auto"]
        )
        
        # Handle different model types
        if self.model_type == "mlm":
            # ESM-style masked language modeling
            # Use the DataCollatorForLanguageModeling to handle the masking
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
            
        elif self.model_type == "clm":
            # ProGen2-style causal/autoregressive language modeling
            # For CLM, we don't need to shift - the model will handle that internally
            # We just need to set up the labels correctly
            batch["labels"] = batch["input_ids"].clone()
            
            # Set padding tokens to -100 so they're ignored in loss calculation
            if self.tokenizer.pad_token_id is not None:
                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
                
        elif self.model_type == "auto":
            # Automatically choose based on sequence properties or randomize
            # Here we'll implement a basic version that randomly chooses between MLM and CLM
            if torch.rand(1).item() > 0.5:
                # Apply MLM
                special_tokens_mask = batch.pop("special_tokens_mask", None)
                batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
            else:
                # Apply CLM
                batch["labels"] = batch["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch
    
    def set_model_type(self, model_type: str) -> None:
        """Dynamically change the model type during training if needed."""
        if model_type not in ["mlm", "clm", "auto"]:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from 'mlm', 'clm', or 'auto'")
        self.model_type = model_type

        
@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Custom TrainingArguments with defaults tailored for our fine-tuning task.
    Extends Hugging Face's TrainingArguments.
    """
    output_dir: str = field(
        default="test_run",
        metadata={"help": "Output directory for model predictions and checkpoints."},
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Training batch size per device."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Evaluation batch size per device."},
    )
    num_train_epochs: float = field(
        default=20,
        metadata={"help": "Total number of training epochs."},
    )
    learning_rate: float = field(
        default=4e-4,
        metadata={"help": "Initial learning rate for Adam."},
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "Number of warmup steps."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay factor."},
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Steps between evaluations."},
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Total number of checkpoints to save."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Checkpoint saving strategy."},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end of training."},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use mixed precision training."},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of data loader workers."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset."},
    )


@dataclass
class TrainingConfig:
    """
    Configuration for fine-tuning the ESM model.
    Contains paths to data, model configuration, training hyperparameters, and wandb settings.
    """
    train_path: str = field(metadata={"help": "Path to training data (FASTA format)."})
    eval_path: str = field(metadata={"help": "Path to validation data (FASTA format)."})
    training_args: Union[dict, CustomTrainingArguments] = field(
        default_factory=CustomTrainingArguments,
        metadata={"help": "Training arguments."},
    )
    base_model: str = field(
        default="facebook/esm2_t6_8M_UR50D",
        metadata={"help": "Base model identifier."},
    )
    wandb_project: str = field(
        default="",
        metadata={"help": "WandB project name. Leave empty to disable WandB."},
    )

    def __post_init__(self) -> None:
        # Convert training_args to CustomTrainingArguments if it's a dictionary
        if isinstance(self.training_args, dict):
            self.training_args = CustomTrainingArguments(**self.training_args)
        
        output_dir = Path(self.training_args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize WandB if a project is specified and on the main process (for multi-GPU setups)
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ["WANDB_PROJECT"] = self.wandb_project
            wandb.init(dir=str(output_dir), group=output_dir.name)
            # Use asdict on self directly since all fields are now proper dataclass instances
            wandb.config.update({"train_config": asdict(self)}, allow_val_change=True)

        # Configure Hugging Face Trainer to report metrics to WandB if enabled
        self.training_args.report_to = ["wandb"] if self.wandb_project else []
        config_path = output_dir / "train_config.yaml"
        with open(config_path, "w") as fp:
            yaml.dump(asdict(self), fp)
        logger.info(f"Training configuration saved to {config_path}")

class ClearEvalMemoryTrainer(Trainer):
    """
    Custom Trainer that clears CUDA cache before evaluation.
    Helps reduce out-of-memory (OOM) errors during evaluation.
    """
    def _clear_cuda_cache(self) -> None:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self._clear_cuda_cache()  # Clear cache before evaluation
        return super().evaluate(*args, **kwargs)


def main() -> None:
    """Main function to fine-tune the ESM model with LoRA integration and WandB logging."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config) as fp:
        config_dict = yaml.safe_load(fp)
    config = TrainingConfig(**config_dict)

    # Load model and tokenizer based on the base model identifier
    if "esm" in config.base_model:
        # Load ESM model and tokenizer for masked language modeling
        model = EsmForMaskedLM.from_pretrained(config.base_model)
        tokenizer = EsmTokenizer.from_pretrained(config.base_model)
        
        # First, ensure all parameters require gradients before applying LoRA
        for param in model.parameters():
            param.requires_grad = True
        
        # Log target modules in the model
        target_modules_found = []
        for name, _ in model.named_modules():
            if any(target in name for target in ["key", "value"]):
                target_modules_found.append(name)
                
        if not target_modules_found:
            logger.warning("No key/value modules found with exact names. Falling back to partial name matching.")
            # Look for modules that might contain key/value in their full path
            for name, _ in model.named_modules():
                if "attention" in name.lower() and ("key" in name.lower() or "value" in name.lower()):
                    target_modules_found.append(name)
        
        if target_modules_found:
            logger.info(f"Found potential target modules: {target_modules_found}")
            # Extract the correct target module names (last part of the full path)
            target_module_names = []
            for full_path in target_modules_found:
                # Extract just the final component name
                parts = full_path.split('.')
                if len(parts) > 0 and any(target in parts[-1] for target in ["key", "value"]):
                    target_module_names.append(parts[-1])
            
            # Remove duplicates
            target_module_names = list(set(target_module_names))
            if target_module_names:
                logger.info(f"Using target module names: {target_module_names}")
            else:
                # If we still don't have good target names, fall back to defaults
                target_module_names = ["key", "value"]
                logger.warning(f"Could not determine exact module names, using defaults: {target_module_names}")
        else:
            # No modules found, use default names as a last resort
            target_module_names = ["key", "value"]
            logger.warning(f"No modules containing key/value found. Using default names: {target_module_names}")
            
        # Verify parameters that will be trainable
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters requiring gradients before LoRA: {trainable_before}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM as the most compatible option for ESM
            r=8,
            lora_alpha=32,
            target_modules=target_module_names,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],  # Also fine-tune the LM head
        )
        
        # Convert to PEFT model
        model = get_peft_model(model, lora_config)
        #model.print_trainable_parameters()  # This is a PEFT utility to summarize trainable params
        
        # Ensure we're only training LoRA and LM head parameters
        for name, param in model.named_parameters():
            if "lora" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
            # Debug: log all trainable parameter names
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name} (shape: {param.shape})")
        
        # Final verification of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA integration complete. Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")
        
        # Safety check
        if trainable_params == 0:
            raise ValueError("No trainable parameters found after LoRA setup!")
            
    elif "progen" in config.base_model:
        # If using a progen model, load it using AutoModel classes with trust_remote_code enabled.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        config_dict = {
            "lora_alpha": 16, 
            "lora_dropout": 0.1,
            "target_modules": ["qkv_proj"],
            "task_type": TaskType.CAUSAL_LM,
            "r": 8
        }

        # Make sure we're in training mode before applying LoRA
        model.train()

        LoRAconfig = LoraConfig(
            r=config_dict["r"],
            inference_mode=False,
            task_type=config_dict["task_type"],
            lora_alpha=config_dict["lora_alpha"],
            lora_dropout=config_dict["lora_dropout"],
            target_modules=config_dict["target_modules"],
        )

        model = get_peft_model(model, LoRAconfig)
        
        # Explicitly set requires_grad for all parameters
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA integration complete for the ProGen model. Trainable: {trainable_params} ({trainable_params/total_params:.2%} of total)")
        
        # Safety check
        if trainable_params == 0:
            raise ValueError("No trainable parameters found after LoRA setup! Training cannot proceed.")
    else:
        raise ValueError(f"Unsupported base model: {config.base_model}")

    # Ensure that the tokenizer's max length matches the model's configuration
    tokenizer.model_max_length = getattr(model.config, "max_position_embeddings", 1024)

    # Load training and evaluation sequences from FASTA files
    train_sequences = [seq.sequence for seq in read_fasta(config.train_path)]
    eval_sequences = [seq.sequence for seq in read_fasta(config.eval_path)]
    logger.info(f"Loaded {len(train_sequences)} training and {len(eval_sequences)} evaluation sequences.")

    # Create custom datasets from the loaded sequences
    train_dataset = SequenceDataset(train_sequences)
    eval_dataset = SequenceDataset(eval_sequences)

    # Initialize the appropriate data collator
    if "esm" in config.base_model:
        # Adjust max_length to be a multiple of pad_to_multiple_of
        max_length = model.config.max_position_embeddings
        pad_to_multiple_of = 8 if config.training_args.fp16 else None
        
        if pad_to_multiple_of and max_length % pad_to_multiple_of != 0:
            max_length = (max_length // pad_to_multiple_of) * pad_to_multiple_of
            logger.info(f"Adjusted max_length to {max_length} to be a multiple of {pad_to_multiple_of}")
        
        data_collator = HybridDataCollator(
            tokenizer=tokenizer,
            model_type="mlm",  # Use masked language modeling for ESM
            mlm_probability=0.15,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        logger.info("Using masked language modeling (MLM) data collator for ESM model.")
        
    elif "progen" in config.base_model:
        # For ProGen models, use CLM mode
        data_collator = HybridDataCollator(
            tokenizer=tokenizer,
            model_type="clm",  # Use causal language modeling for ProGen
            max_length=tokenizer.model_max_length,
            pad_to_multiple_of=8 if config.training_args.fp16 else None,
        )
        logger.info("Using causal language modeling (CLM) data collator for ProGen model.")

    # Verify model is in training mode with trainable parameters
    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters before training: {trainable_params}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Check parameter freezing logic.")

    # Create a Trainer instance with our custom ClearEvalMemoryTrainer to handle evaluation memory
    trainer = ClearEvalMemoryTrainer(
        model=model,
        args=config.training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Check for an existing checkpoint to resume training from, if available
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {checkpoint}")

    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Save the final model

    # Log the artifact
    artifact = wandb.Artifact("esm2-lora-finetuned", type="model")
    artifact.add_dir(config.training_args.output_dir)
    wandb.log_artifact(artifact)
    
    # Log training metrics
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    logger.info("Training complete.")
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
    # Log evaluation metrics
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    print("Evaluation metrics:", eval_metrics)
    
    # Now finish the WandB run
    wandb.finish()
    
if __name__ == "__main__":
    main()