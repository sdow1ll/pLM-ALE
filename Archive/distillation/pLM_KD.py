#!/usr/bin/env python
"""Knowledge distillation from a fine-tuned ESM2 model to a smaller ESM2 model with WandB logging."""

import sys
sys.path.append('/home/sdowell/scratch/Thesis/BenchmarkingFinetuning')
from finetuneESM2_ProGen2_LoRA import *

# Override or add additional imports if needed
import torch.nn.functional as F


@dataclass
class DistillationTrainingArguments(CustomTrainingArguments):
    """Extends the CustomTrainingArguments with knowledge distillation specific parameters."""
    alpha: float = field(
        default=0.7,
        metadata={"help": "Weight for soft targets (teacher predictions) in distillation loss."},
    )
    temperature: float = field(
        default=2.5,
        metadata={"help": "Temperature for softening probability distributions in distillation."},
    )


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_model_path: str
    student_model_path: str
    train_path: str
    eval_path: str
    base_model: str = "facebook/esm2_t6_8M_UR50D"
    wandb_project: str = ""
    wandb_run_name: str = "esm2_knowledge_distillation"
    training_args: Union[dict, DistillationTrainingArguments] = field(
        default_factory=DistillationTrainingArguments
    )
    use_lora: bool = False
    
    def __post_init__(self):
        # Convert training_args to DistillationTrainingArguments if it's a dictionary
        if isinstance(self.training_args, dict):
            self.training_args = DistillationTrainingArguments(**self.training_args)
        
        output_dir = Path(self.training_args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize WandB if specified
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ["WANDB_PROJECT"] = self.wandb_project
            wandb.init(dir=str(output_dir), name=self.wandb_run_name)
            # Use asdict on self directly
            wandb.config.update({"distill_config": asdict(self)}, allow_val_change=True)

        # Configure Hugging Face Trainer to report metrics to WandB if enabled
        self.training_args.report_to = ["wandb"] if self.wandb_project else []
        
        # Save configuration
        config_path = output_dir / "distillation_config.yaml"
        with open(config_path, "w") as fp:
            yaml.dump(asdict(self), fp)
        logger.info(f"Distillation configuration saved to {config_path}")


class DistillationTrainer(ClearEvalMemoryTrainer):
    """
    Custom Trainer that implements knowledge distillation from a teacher model to a student model.
    Extends the ClearEvalMemoryTrainer from the base script.
    """
    def __init__(self, teacher_model=None, alpha=0.7, temperature=2.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Make sure teacher model is in eval mode
        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to implement knowledge distillation.
        This avoids issues with the compute_loss signature.
        """
        model.train()
        
        # Extract labels that were created by the data collator
        labels = inputs.pop("labels", None)
        
        # Forward pass through student model
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Forward pass through teacher model (no gradient tracking needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Calculate task-specific loss (based on ground truth labels)
        task_loss = 0
        
        if labels is not None:
            # For masked language modeling, only compute task loss on masked tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            mask = (labels != -100)
            if mask.sum() > 0:
                task_loss = loss_fct(student_logits[mask], labels[mask])
        
        # Calculate distillation loss (KL divergence from teacher predictions)
        # Apply temperature scaling for softer probabilities
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        distill_loss = F.kl_div(
            input=soft_prob,
            target=soft_targets,
            reduction='mean'
        ) * (self.temperature ** 2)  # Scale by T^2 as in the original paper (2015)
        
        # Weighted sum of the two losses
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        
        # Log both components during training
        if self.args.local_process_index == 0 and self.state.global_step % 10 == 0:
            if wandb.run is not None:
                wandb.log({
                    "train/train_loss": total_loss.item(),
                    "train/task_loss": task_loss.item(),
                    "train/distill_loss": distill_loss.item(),
                    "train/alpha": self.alpha,
                    "train/temperature": self.temperature
                }, step=self.state.global_step)
        
        # Scale the loss if we're doing gradient accumulation
        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            total_loss = total_loss / num_items_in_batch
            
        return total_loss


def main_distillation() -> None:
    """Main function to run knowledge distillation from a fine-tuned ESM2 model to a smaller one."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config) as fp:
        config_dict = yaml.safe_load(fp)
    config = DistillationConfig(**config_dict)

    # Load teacher model (the fine-tuned ESM2 model)
    logger.info(f"Loading teacher model from {config.teacher_model_path}")
    teacher_model = EsmForMaskedLM.from_pretrained(config.teacher_model_path)
    
    # For the tokenizer, use the base ESM2 tokenizer since it may not be in the fine-tuned model
    teacher_base_model = "facebook/esm2_t30_150M_UR50D"  # Adjust based on your teacher model
    tokenizer = EsmTokenizer.from_pretrained(teacher_base_model)
    
    # Load student model 
    logger.info(f"Loading student model from {config.base_model}")
    student_model = EsmForMaskedLM.from_pretrained(config.base_model)
    
    # Apply LoRA to student model if specified
    if hasattr(config, 'use_lora') and config.use_lora:
        logger.info("Applying LoRA to student model")
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for ESM compatibility
            r=8,
            lora_alpha=32,
            target_modules=["key", "value"],  # Adjust based on your model architecture
            lora_dropout=0.1,
            bias="none"
        )
        
        # Apply LoRA to student model
        student_model = get_peft_model(student_model, lora_config)
    
    # Set tokenizer max length
    max_length = min(
        getattr(teacher_model.config, "max_position_embeddings", 1024),
        getattr(student_model.config, "max_position_embeddings", 1024)
    )
    tokenizer.model_max_length = max_length
    
    # Load sequences from FASTA file (no labels needed)
    train_sequences = [seq.sequence for seq in read_fasta(config.train_path)]
    eval_sequences = [seq.sequence for seq in read_fasta(config.eval_path)]
    logger.info(f"Loaded {len(train_sequences)} training and {len(eval_sequences)} evaluation sequences.")

    # Create custom datasets from the loaded sequences
    train_dataset = SequenceDataset(train_sequences)
    eval_dataset = SequenceDataset(eval_sequences)

    # Initialize data collator for masked language modeling
    # This will AUTOMATICALLY create masked tokens and labels during training
    pad_to_multiple_of = 8 if config.training_args.fp16 else None
    
    data_collator = HybridDataCollator(
        tokenizer=tokenizer,
        model_type="mlm",  # Use masked language modeling for ESM
        mlm_probability=0.15,  # 15% of tokens will be masked
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of
    )
    
    # Move teacher model to the same device that will be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    teacher_model.eval()  # Ensure teacher is in evaluation mode
    
    # Create a DistillationTrainer instance
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=config.training_args.alpha,
        temperature=config.training_args.temperature,
        model=student_model,
        args=config.training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Check for an existing checkpoint to resume training from
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {checkpoint}")

    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Save the final model

    # Log the artifact to wandb if enabled
    if config.wandb_project:
        artifact = wandb.Artifact("esm2-distilled", type="model")
        artifact.add_dir(config.training_args.output_dir)
        wandb.log_artifact(artifact)
    
    # Log training metrics
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    logger.info("Knowledge distillation training complete.")
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
    # Log evaluation metrics
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    print("Evaluation metrics:", eval_metrics)
    
    # Finish the WandB run if it was started
    if config.wandb_project and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main_distillation()
    