"""Finetune the ESM model."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from argparse import ArgumentParser
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Union
import torch
import transformers
import wandb
import yaml
from torch.utils.data import Dataset
from transformers import EsmForMaskedLM, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import EsmTokenizer
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import EarlyStoppingCallback

#PathLike = str | Path #python 3.10
PathLike = Union[str, Path] # python 3.9


@dataclass
class Sequence:
    """Store a biological sequence and its description tag."""

    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""

    def __hash__(self) -> int:
        """Hash the sequence and tag."""
        return hash((self.sequence, self.tag))


def read_fasta(fasta_file: PathLike) -> list[Sequence]:
    """Read fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: PathLike,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        f.write('\n'.join(f'>{seq.tag}\n{seq.sequence}' for seq in seqs))


def random_split_fasta(
    input_fasta: PathLike,
    output_dir: PathLike,
    split: float = 0.8,
    seed: int = 0,
) -> None:
    """Randomly split a fasta file into train and validation fasta file."""
    # Read the input file
    sequences = read_fasta(input_fasta)

    # Shuffle the sequences
    random.seed(seed)
    random.shuffle(sequences)

    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write the train and validation fasta files
    split_idx = int(len(sequences) * split)
    write_fasta(sequences[:split_idx], output_dir / 'train.fasta')
    write_fasta(sequences[split_idx:], output_dir / 'valid.fasta')

    # Copy the original fasta file to the output directory for reference
    shutil.copy(input_fasta, output_dir)

    # Log JSON metadata on the split
    metadata = {
        'input_fasta': str(Path(input_fasta).resolve()),
        'output_dir': str(output_dir.resolve()),
        'split': split,
        'seed': seed,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Write the metadata to a JSON file
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


class SequenceDataset(Dataset):
    """Dataset for sequences."""

    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """Return the idx'th sequence."""
        # Get the idx'th sequence
        return self.sequences[idx]


class DataCollator(transformers.DataCollatorForLanguageModeling):
    """Data collator for language modeling with ESM."""

    def __init__(self, train_mode: bool = False, **kwargs: Any) -> None:
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def tokenize(self, sequences: list[str]) -> transformers.BatchEncoding:
        """Tokenize the sequences and return a BatchEncoding."""
        return self.tokenizer(
            sequences,
            return_tensors='pt',
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode and self.mlm,
        )

    def torch_call(self, examples: list[str]) -> transformers.BatchEncoding:
        """Tokenize the batch and prepare the input and labels."""
        # First, tokenize the batch
        batch = self.tokenize(examples)

        # We only need to mask tokens if we are doing inference
        if not self.train_mode:
            return batch

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop('special_tokens_mask', None)

        if self.mlm:
            batch['input_ids'], batch['labels'] = self.torch_mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels
        return batch


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """TrainingArguments for configuring the Hugging Face Trainer.

    Here we provide some sensible defaults for the arguments for our use case.
    """

    output_dir: str = field(
        default='test_run',
        metadata={
            'help': 'The output directory where the model predictions and '
            'checkpoints will be written.'
        },
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'},
    )
    num_train_epochs: float = field(
        default=20,
        metadata={'help': 'Total number of training epochs to perform.'},
    )
    learning_rate: float = field(
        default=4e-4,
        metadata={'help': 'The initial learning rate for Adam.'},
    )
    warmup_steps: int = field(
        default=1_000,
        metadata={'help': 'Linear warmup over `warmup_steps`.'},
    )
    lr_scheduler_type: str = field(
        default='cosine',
        metadata={'help': 'The scheduler type to use.'},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={'help': 'The weight decay to apply.'},
    )
    eval_steps: int = field(
        default=500,
        metadata={
            'help': 'Number of steps between evaluations. If `eval_steps` '
            'is modified, update `logging_steps` and `save_steps` to the same '
            'value.'
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={'help': 'Total number of checkpoints to save.'},
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for saving checkpoints.'},
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for evaluating.'},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            'help': 'Whether to load the best model at the end of training. '
            'When `save_total_limit` is set to 1, will save the best model as '
            'well as the last model if the last model is worse (eval_loss) '
            'than the best model.'
        },
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Whether to use 16-bit (mixed) precision training.'},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={'help': 'Number of subprocesses to use for data loading.'},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            'help': 'This skips underlying logic in Trainer which modifies '
            'the data_collator (do not change).'
        },
    )
    # Added additional fields for improved control
    run_name: str = field(
        default="",
        metadata={'help': 'Name for the wandb run.'},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={'help': 'Max gradient norm for gradient clipping.'},
    )
    logging_steps: int = field(
        default=50,
        metadata={'help': 'Log every X updates steps.'},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={'help': 'Metric to use for determining the best model.'},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={'help': 'Whether higher is better for the metric.'},
    )


@dataclass
class TrainingConfig:
    """Configuration for fine tuning the ESM model."""

    train_path: str = field(
        metadata={'help': 'Path to training data.'},
    )
    eval_path: str = field(
        metadata={'help': 'Path to validation data.'},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={
            'help': 'Hugging face arguments for training the model '
            '(see transformers.TrainingArguments).'
        },
    )
    base_model: str = field(
        default='facebook/esm2_t6_8M_UR50D',
        metadata={'help': 'Base model to use for training.'},
    )
    wandb_project: str = field(
        default='',
        metadata={
            'help': 'Wandb project name (By default, set to empty string'
            ' to turn off wandb).'
        },
    )
    # New transfer learning parameters
    num_layers_to_unfreeze: int = field(
        default=2,
        metadata={'help': 'Number of transformer layers to unfreeze for fine-tuning.'},
    )
    unfreeze_embeddings: bool = field(
        default=False,
        metadata={'help': 'Whether to unfreeze the embedding layer.'},
    )
    # Model architecture parameters
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={'help': 'Dropout probability for hidden layers.'},
    )
    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={'help': 'Dropout probability for attention probabilities.'},
    )
    # Data preparation parameters
    mlm_probability: float = field(
        default=0.15,
        metadata={'help': 'Probability of masking tokens for masked language modeling.'},
    )
    # Early stopping parameters
    early_stopping_patience: int = field(
        default=3,
        metadata={'help': 'Number of evaluations with no improvement after which training will be stopped.'},
    )
    early_stopping_threshold: float = field(
        default=0.001,
        metadata={'help': 'Minimum change to qualify as improvement for early stopping.'},
    )
    # Logging and monitoring
    log_model_weights: bool = field(
        default=True,
        metadata={'help': 'Whether to log model weight statistics during training.'},
    )
    wandb_tags: list[str] = field(
        default_factory=list,
        metadata={'help': 'List of tags for wandb run.'},
    )
    wandb_notes: str = field(
        default='',
        metadata={'help': 'Notes for wandb run.'},
    )

    def __post_init__(self) -> None:
        """Initialize the training arguments and log the config."""
        # Populate the training arguments
        self.training_args = TrainingArguments(**self.training_args)

        # Set the output directory
        output_dir = Path(self.training_args.output_dir)

        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)

        # wandb needs to be initialized once on all node ranks
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ['WANDB_PROJECT'] = self.wandb_project
            # Assign the same group name as the output directory
            # so that multi-node runs are grouped together
            wandb.init(
                dir=output_dir, 
                group=output_dir.name,
                tags=self.wandb_tags,
                notes=self.wandb_notes,
                name=self.training_args.run_name if hasattr(self.training_args, 'run_name') and self.training_args.run_name else None
            )
            wandb.config.update(
                {'train_config': asdict(self)}, allow_val_change=True
            )

        # Set the report_to argument based on the wandb project
        self.training_args.report_to = ['wandb' if self.wandb_project else '']

        # Log the config to a yaml file
        with open(output_dir / 'train_config.yaml', 'w') as fp:
            yaml.dump(asdict(self), fp)


class ClearEvalMemoryTrainer(Trainer):
    """Trainer that clears the cuda cache before each evaluation.

    Note: reduces OOMs for some models.
    """

    def _clear_cuda_cache(self) -> None:
        import gc

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()

    def training_step(self, *args: Any, **kwargs: Any) -> float:
        """Run training step and periodically clear cache."""
        loss = super().training_step(*args, **kwargs)
        # Clear cache every 50 steps to prevent gradual memory buildup
        if self.state.global_step % 50 == 0:
            self._clear_cuda_cache()
        return loss
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Clear the cuda cache before evaluation."""
        self._clear_cuda_cache()
        return super().evaluate(*args, **kwargs)


from safetensors.torch import load_file  # Import SafeTensors loader

def main() -> None:
    """Fine-tune the ESM model."""
    # Parse YAML file to get the training config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as fp:
        config = TrainingConfig(**yaml.safe_load(fp))

    # Load model and tokenizer
    if "esm" in config.base_model:
        # Load with custom dropout probabilities
        model_config = AutoConfig.from_pretrained(
            config.base_model,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        model = EsmForMaskedLM.from_pretrained(config.base_model, config=model_config)
    # Memory optimization: Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled for memory optimization")
        tokenizer = EsmTokenizer.from_pretrained(config.base_model)
    elif "progen" in config.base_model:
        model_config = AutoConfig.from_pretrained(
            config.base_model,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, config=model_config, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True
        )
    else:
        raise ValueError(f"Unsupported base model: {config.base_model}")

    # --- IMPROVED TRANSFER LEARNING IMPLEMENTATION ---
    
    # First, freeze all parameters in the model (both base model and head)
    for param in model.parameters():
        param.requires_grad = False
    
    # Now, selectively unfreeze components for transfer learning
    
    # 1. Always unfreeze the language modeling head for prediction
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        print("Unfrozen: Language modeling head (lm_head)")
    
    # 2. Optionally unfreeze the embedding layer
    if config.unfreeze_embeddings and hasattr(model, "esm") and hasattr(model.esm, "embeddings"):
        for param in model.esm.embeddings.parameters():
            param.requires_grad = True
        print("Unfrozen: Embedding layer")
    
    # 3. Unfreeze the specified number of transformer layers
    if "esm" in config.base_model:
        if hasattr(model, "esm") and hasattr(model.esm, "encoder") and hasattr(model.esm.encoder, "layer"):
            total_layers = len(model.esm.encoder.layer)
            
            # Ensure num_layers_to_unfreeze is not more than total_layers
            num_layers_to_unfreeze = min(config.num_layers_to_unfreeze, total_layers)
            
            # Unfreeze the last N layers
            for i in range(total_layers - num_layers_to_unfreeze, total_layers):
                for param in model.esm.encoder.layer[i].parameters():
                    param.requires_grad = True
                print(f"Unfrozen: Transformer layer {i} of {total_layers-1}")
    
    # Print summary of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} of {total_params:,} ({trainable_params/total_params:.2%})")
    
    # --- END OF IMPROVED TRANSFER LEARNING IMPLEMENTATION ---

    # Print which parameters are trainable for verification
    print("\nDetailed trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={param.shape}, requires_grad=True")

    # Set tokenizer max length
    tokenizer.model_max_length = model.config.max_position_embeddings

    # Prepare datasets
    train_sequences = [seq.sequence for seq in read_fasta(config.train_path)]
    eval_sequences = [seq.sequence for seq in read_fasta(config.eval_path)]
    train_dataset = SequenceDataset(train_sequences)
    eval_dataset = SequenceDataset(eval_sequences)

    # Create data collator with configurable MLM probability
    data_collator = DataCollator(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
        pad_to_multiple_of=8 if config.training_args.fp16 else None,
    )

    # Create callbacks including early stopping
    callbacks = []
    
    # Add early stopping callback
    if config.training_args.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        )
    
    # Add loss monitoring callback
    class LossMonitorCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                current_loss = logs["loss"]
                print(f"Step {state.global_step}: Training loss = {current_loss:.4f}")
                
                # Alert if loss is not decreasing after initial steps
                if state.global_step > 100 and current_loss > 0.95 * logs.get("loss", 0):
                    print("WARNING: Loss not decreasing significantly. Check if model is learning properly.")
    
    callbacks.append(LossMonitorCallback())
    
    # Add weight monitoring callback if enabled
    if config.log_model_weights and config.wandb_project:
        class WeightMonitorCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                if hasattr(model, "esm") and hasattr(model.esm, "embeddings"):
                    emb_weight = model.esm.embeddings.word_embeddings.weight
                    wandb.log({
                        "embedding_mean": emb_weight.mean().item(),
                        "embedding_std": emb_weight.std().item()
                    }, step=state.global_step)
                
                if hasattr(model, "lm_head"):
                    if hasattr(model.lm_head, "decoder") and hasattr(model.lm_head.decoder, "weight"):
                        head_weight = model.lm_head.decoder.weight
                    elif hasattr(model.lm_head, "weight"):
                        head_weight = model.lm_head.weight
                    else:
                        return
                        
                    wandb.log({
                        "lm_head_mean": head_weight.mean().item(),
                        "lm_head_std": head_weight.std().item()
                    }, step=state.global_step)
        
        callbacks.append(WeightMonitorCallback())

    # Initialize Trainer
    trainer = ClearEvalMemoryTrainer(
        model=model,
        args=config.training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )

    # Attempt to load a checkpoint
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint:
        safetensors_path = f"{checkpoint}/model.safetensors"
        if os.path.exists(safetensors_path):
            print(f"Resuming training from checkpoint: {checkpoint}")
            
            # Load model weights from SafeTensors file
            model_state = load_file(safetensors_path, device="cpu")
            
            # IMPORTANT FIX: Don't initialize missing keys with zeros
            # Instead, log which keys are missing but keep their original initialization
            missing_keys = []
            for key in model.state_dict().keys():
                if key not in model_state:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys missing from checkpoint. First few: {missing_keys[:5]}")
                print("These parameters will retain their initialization values.")
            
            # Load state dict with strict=False to handle missing keys
            trainer.model.load_state_dict(model_state, strict=False)
            
            # Reinitialize the optimizer to match current trainable parameters
            trainer.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=config.training_args.learning_rate,
            )
        else:
            print(f"Checkpoint found, but {safetensors_path} is missing. Starting fresh.")

    # Train the model
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Check model weights after training
    def check_model_weights():
        if hasattr(model, "esm") and hasattr(model.esm, "embeddings"):
            emb_weight = model.esm.embeddings.word_embeddings.weight
            print(f"\nAfter training - Embedding stats: Mean={emb_weight.mean().item():.6f}, Std={emb_weight.std().item():.6f}")
        
        if hasattr(model, "lm_head"):
            if hasattr(model.lm_head, "decoder") and hasattr(model.lm_head.decoder, "weight"):
                head_weight = model.lm_head.decoder.weight
            elif hasattr(model.lm_head, "weight"):
                head_weight = model.lm_head.weight
            else:
                print("Could not locate lm_head weights")
                return
                
            print(f"After training - LM head stats: Mean={head_weight.mean().item():.6f}, Std={head_weight.std().item():.6f}")
            
            # If the head weights are all zeros or very close to zero, something is wrong
            if head_weight.abs().mean().item() < 1e-6:
                print("WARNING: LM head weights are all close to zero. Fine-tuning may not have worked correctly.")
    
    # Save the model and metrics
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    
    # Check weights after saving
    check_model_weights()
    
    print("\nTraining complete! Model saved to:", config.training_args.output_dir)

if __name__ == '__main__':
    main()