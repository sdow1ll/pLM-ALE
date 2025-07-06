import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import argparse
from tqdm import tqdm
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging
import json
import pandas as pd
import wandb
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProteinDataset(Dataset):
    """Dataset for unsupervised learning on protein sequences."""
    
    def __init__(self, sequences, tokenizer, max_length=1024, mlm_probability=0.15):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of protein sequences
            tokenizer: ESM tokenizer
            max_length: Maximum sequence length
            mlm_probability: Probability of masking tokens for MLM
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For MLM: create masked inputs
        mlm_input_ids, mlm_labels = self.mask_tokens(input_ids.clone())
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_input_ids": mlm_input_ids,
            "mlm_labels": mlm_labels
        }
    
    def mask_tokens(self, input_ids):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Tuple of masked input IDs and labels
        """
        labels = input_ids.clone()
        
        # We sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels, already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # Don't mask padding tokens
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Mask tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%), keep the original token
        
        return input_ids, labels


def load_fasta(fasta_file: str) -> List[str]:
    """
    Load sequences from a FASTA file.
    
    Args:
        fasta_file: Path to the FASTA file
        
    Returns:
        List of sequences
    """
    sequences = []
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save the previous sequence if it exists
                if current_seq:
                    sequences.append(''.join(current_seq))
                
                # Start a new sequence
                current_seq = []
            else:
                # Add to the current sequence
                current_seq.append(line)
    
    # Save the last sequence
    if current_seq:
        sequences.append(''.join(current_seq))
    
    return sequences


def load_esm2_model(model_name="facebook/esm2_t33_650M_UR50D"):
    """Load ESM-2 model and tokenizer for masked language modeling."""
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def add_lora_to_model(model, lora_rank=16, lora_alpha=32, lora_dropout=0.05):
    """Add LoRA adapters to the model using a targeted approach for ESM models."""
    import re
    
    # Print total number of parameters before applying LoRA
    total_original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_original_params:,} total parameters before LoRA")
    
    # Find modules that might be attention components
    all_modules = dict(model.named_modules())
    logger.info(f"Total modules: {len(all_modules)}")
    
    # Target all linear modules (which is a safe fallback)
    linear_modules = []
    for name, module in all_modules.items():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append(name)
    
    logger.info(f"Found {len(linear_modules)} Linear modules")
    if len(linear_modules) > 0:
        logger.info(f"Example linear modules: {linear_modules[:5]}")
    
    # Target a subset of linear modules (to avoid too many parameters)
    # Pattern matching for attention modules
    attention_pattern = re.compile(r'(attn|attention|query|key|value|q_proj|k_proj|v_proj|qkv)')
    attn_modules = [name for name in linear_modules if attention_pattern.search(name.lower())]
    
    if len(attn_modules) > 0:
        logger.info(f"Found {len(attn_modules)} attention-related modules: {attn_modules[:10]}")
        target_modules = attn_modules
    else:
        # Target a subset of linear modules
        logger.info("No attention modules found by pattern matching, using selected linear modules")
        # Pick a subset of linear modules, especially those in the encoder layers
        encoder_linears = [name for name in linear_modules if 'encoder' in name.lower()]
        if len(encoder_linears) > 0:
            target_modules = encoder_linears[:10]  # Limit to 10 to be safe
        else:
            target_modules = linear_modules[:10]  # Limit to 10 to be safe
    
    logger.info(f"Selected target modules: {target_modules}")
    
    # For ESM-2, try a single, known module as a fallback
    if not target_modules and hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        logger.info("Using direct attribute access to find attention modules")
        # Sample a few names to be safe
        target_modules = ["encoder.layer.0.intermediate.dense", "encoder.layer.0.output.dense"]
        logger.info(f"Using fallback target modules: {target_modules}")
    
    # Final fallback - just specify general module names for PEFT to find
    if not target_modules:
        logger.info("Using generic module names as fallback")
        target_modules = ["dense", "linear"]
    
    # Configure LoRA with found target modules
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Using CAUSAL_LM for MLM
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        # Use bias="none" to avoid bias adapters which can cause issues
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters after LoRA: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model, {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "percent_trainable": trainable_params/total_params * 100
    }


def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batch_losses = []
    
    start_time = time.time()
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        mlm_input_ids = batch["mlm_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        
        # Forward pass for MLM
        outputs = model(
            input_ids=mlm_input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels
        )
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Track loss
        loss_value = loss.item()
        total_loss += loss_value
        batch_losses.append(loss_value)
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
        
        # Log to wandb every 10 steps
        if (step + 1) % 10 == 0:
            wandb.log({
                "train_step_loss": loss_value,
                "train_step": step,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    metrics = {
        "train_loss": avg_loss,
        "train_perplexity": perplexity.item(),
        "train_time_seconds": epoch_time,
        "train_samples_per_second": len(dataloader.dataset) / epoch_time
    }
    
    return avg_loss, metrics


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            mlm_input_ids = batch["mlm_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            
            # Forward pass for MLM
            outputs = model(
                input_ids=mlm_input_ids,
                attention_mask=attention_mask,
                labels=mlm_labels
            )
            loss = outputs.loss
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    eval_time = time.time() - start_time
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity.item(),
        "eval_time_seconds": eval_time,
        "eval_samples_per_second": len(dataloader.dataset) / eval_time
    }
    
    return avg_loss, metrics


def extract_embeddings(model, dataloader, device, layer=-1):
    """
    Extract embeddings from the model for all sequences in the dataloader.
    
    Args:
        model: ESM-2 model
        dataloader: DataLoader for sequences
        device: Device to use
        layer: Which layer to extract embeddings from (-1 for last layer)
        
    Returns:
        Numpy array of embeddings
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states from the specified layer
            hidden_states = outputs.hidden_states[layer]
            
            # Get CLS token embedding or mean of sequence embeddings
            # Option 1: CLS token (first token)
            # embeddings = hidden_states[:, 0, :].cpu().numpy()
            
            # Option 2: Mean of all tokens (excluding padding)
            # Create a mask that is 1 for real tokens and 0 for padding tokens
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # Sum the embeddings of all tokens and divide by the number of tokens
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            count_tokens = torch.sum(mask, dim=1)
            embeddings = (sum_embeddings / count_tokens).cpu().numpy()
            
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)


def cluster_embeddings(embeddings, n_clusters=10):
    """
    Cluster embeddings using K-means.
    
    Args:
        embeddings: Numpy array of embeddings
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (cluster assignments, cluster centers, reduced embeddings)
    """
    # Reduce dimensionality for visualization and clustering
    logger.info(f"Performing PCA dimensionality reduction...")
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Log explained variance to wandb
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    variance_data = [[i+1, var, cum_var] for i, (var, cum_var) in 
                    enumerate(zip(explained_variance[:20], cumulative_variance[:20]))]
    
    variance_table = wandb.Table(
        columns=["Component", "Explained Variance", "Cumulative Variance"],
        data=variance_data
    )
    wandb.log({"pca_variance": variance_table})
    
    # Create elbow plot to determine optimal number of clusters
    if wandb.run is not None:
        inertia_values = []
        k_values = range(2, min(21, n_clusters * 2))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(reduced_embeddings)
            inertia_values.append(kmeans.inertia_)
        
        elbow_data = [[k, inertia] for k, inertia in zip(k_values, inertia_values)]
        elbow_table = wandb.Table(
            columns=["Number of Clusters", "Inertia"],
            data=elbow_data
        )
        wandb.log({"kmeans_elbow_plot": elbow_table})
    
    # Cluster
    logger.info(f"Clustering with K-means (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(reduced_embeddings)
    
    # Log cluster sizes to wandb
    cluster_counts = np.bincount(clusters)
    cluster_sizes = [[i, count] for i, count in enumerate(cluster_counts)]
    cluster_table = wandb.Table(
        columns=["Cluster ID", "Size"],
        data=cluster_sizes
    )
    wandb.log({"cluster_sizes": cluster_table})
    
    return clusters, kmeans.cluster_centers_, reduced_embeddings


def visualize_clusters(reduced_embeddings, clusters, output_file=None):
    """
    Visualize clusters using PCA.
    
    Args:
        reduced_embeddings: Reduced embeddings from PCA
        clusters: Cluster assignments
        output_file: Path to save the visualization (if None, display it)
    """
    # Further reduce to 2D for visualization
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(reduced_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Protein Sequence Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    if output_file:
        plt.savefig(output_file)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({"cluster_visualization": wandb.Image(output_file)})
    else:
        plt.show()
    
    # Also create a 3D visualization if possible
    try:
        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(reduced_embeddings)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            embeddings_3d[:, 0], 
            embeddings_3d[:, 1], 
            embeddings_3d[:, 2],
            c=clusters, 
            cmap='viridis', 
            alpha=0.5
        )
        plt.colorbar(scatter, label='Cluster')
        ax.set_title('Protein Sequence Clusters (3D)')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        
        if output_file:
            output_file_3d = output_file.replace('.png', '_3d.png')
            plt.savefig(output_file_3d)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({"cluster_visualization_3d": wandb.Image(output_file_3d)})
        else:
            plt.show()
    except Exception as e:
        logger.warning(f"Could not create 3D visualization: {e}")


def init_wandb(args):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        args: Command-line arguments
    """
    # Generate a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"esm2_lora_{timestamp}"
    
    # Extract model size from model name
    model_size = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    
    # Create config dictionary for wandb
    config = {
        # Model parameters
        "model_name": args.model_name,
        "model_size": model_size,
        
        # LoRA parameters
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        
        # Training parameters
        "max_length": args.max_length,
        "mlm_probability": args.mlm_probability,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        
        # Data parameters
        "train_data": os.path.basename(args.train_data),
        "val_data": os.path.basename(args.val_data) if args.val_data else None,
        
        # Mode
        "mode": "extract_only" if args.extract_only else "train",
        "extract_after_training": args.extract_after_training,
        "cluster": args.cluster,
        "n_clusters": args.n_clusters,
    }
    
    # Initialize wandb
    wandb.init(
        project="esm2-lora-unsupervised",
        name=run_name,
        config=config,
        mode="online" if not args.disable_wandb else "disabled"
    )
    
    return run_name


def main(args):
    # Initialize wandb
    if not args.disable_wandb:
        run_name = init_wandb(args)
    else:
        run_name = f"esm2_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Load model and tokenizer
    model, tokenizer = load_esm2_model(args.model_name)
    
    # Add LoRA adapters if we're training
    if not args.extract_only:
        model, param_info = add_lora_to_model(
            model, 
            lora_rank=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        
        # Log parameter info to wandb
        if not args.disable_wandb:
            wandb.log(param_info)
    
    model.to(device)
    
    # Log model architecture to wandb
    if not args.disable_wandb:
        # Create a model summary
        model_summary = f"Model: {args.model_name}\n"
        model_summary += f"LoRA rank: {args.lora_rank}\n"
        model_summary += f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n"
        model_summary += f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        
        wandb.log({"model_summary": model_summary})
    
    # Load sequences
    logger.info(f"Loading sequences from {args.train_data}")
    train_sequences = load_fasta(args.train_data)
    logger.info(f"Loaded {len(train_sequences)} training sequences")
    
    # Log sequence length statistics to wandb
    if not args.disable_wandb:
        seq_lengths = [len(seq) for seq in train_sequences]
        seq_stats = {
            "min_length": min(seq_lengths),
            "max_length": max(seq_lengths),
            "mean_length": sum(seq_lengths) / len(seq_lengths),
            "median_length": sorted(seq_lengths)[len(seq_lengths) // 2]
        }
        wandb.log({"sequence_stats": seq_stats})
        
        # Create a histogram of sequence lengths
        plt.figure(figsize=(10, 6))
        plt.hist(seq_lengths, bins=50)
        plt.title('Distribution of Sequence Lengths')
        plt.xlabel('Length')
        plt.ylabel('Count')
        
        if output_dir:
            length_hist_file = os.path.join(output_dir, "sequence_length_histogram.png")
            plt.savefig(length_hist_file)
            wandb.log({"sequence_length_histogram": wandb.Image(length_hist_file)})
        plt.close()
    
    if args.val_data:
        logger.info(f"Loading validation sequences from {args.val_data}")
        val_sequences = load_fasta(args.val_data)
        logger.info(f"Loaded {len(val_sequences)} validation sequences")
    
    # Create datasets and dataloaders
    train_dataset = ProteinDataset(
        train_sequences, 
        tokenizer, 
        max_length=args.max_length,
        mlm_probability=args.mlm_probability
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    if args.val_data:
        val_dataset = ProteinDataset(
            val_sequences, 
            tokenizer, 
            max_length=args.max_length,
            mlm_probability=args.mlm_probability
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size
        )
    
    # If we're only extracting embeddings, do that and exit
    if args.extract_only:
        logger.info("Extract-only mode: extracting embeddings without training")
        
        # Create a simple dataloader for extraction
        extraction_dataset = ProteinDataset(
            train_sequences, 
            tokenizer, 
            max_length=args.max_length,
            mlm_probability=0.0  # No masking for extraction
        )
        
        extraction_dataloader = DataLoader(
            extraction_dataset, 
            batch_size=args.batch_size
        )
        
        # Extract embeddings
        embeddings = extract_embeddings(model, extraction_dataloader, device)
        
        # Save embeddings
        if output_dir:
            embeddings_file = os.path.join(output_dir, "embeddings.npy")
            np.save(embeddings_file, embeddings)
            logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Cluster embeddings
        if args.cluster:
            logger.info(f"Clustering embeddings into {args.n_clusters} clusters")
            clusters, centers, reduced_embeddings = cluster_embeddings(
                embeddings, 
                n_clusters=args.n_clusters
            )
            
            # Save cluster assignments
            if output_dir:
                clusters_file = os.path.join(output_dir, "clusters.npy")
                np.save(clusters_file, clusters)
                logger.info(f"Saved cluster assignments to {clusters_file}")
                
                # Visualize clusters
                viz_file = os.path.join(output_dir, "clusters_visualization.png")
                visualize_clusters(reduced_embeddings, clusters, viz_file)
                logger.info(f"Saved cluster visualization to {viz_file}")
        
        logger.info("Extraction completed")
        
        # Finish the wandb run
        if not args.disable_wandb:
            wandb.finish()
        
        return

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device
        )
        
        # Log training metrics
        if not args.disable_wandb:
            wandb.log({
                "epoch": epoch + 1,
                **train_metrics
            })
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        
        # Evaluate if validation data is provided
        if args.val_data:
            val_loss, val_metrics = evaluate(model, val_dataloader, device)
            
            # Log validation metrics
            if not args.disable_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    **val_metrics
                })
            
            logger.info(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                if output_dir:
                    # Save the model
                    model_path = os.path.join(output_dir, "best_model")
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
                    logger.info(f"Saved best model to {model_path}")
        
        # Save checkpoint
        if output_dir:
            checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            model.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training completed!")
    
    # Save final model
    if output_dir:
        final_model_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

    # Extract embeddings after training
    if args.extract_after_training:
        logger.info("Extracting embeddings after training")
        
        # Create a simple dataloader for extraction
        extraction_dataset = ProteinDataset(
            train_sequences, 
            tokenizer, 
            max_length=args.max_length,
            mlm_probability=0.0  # No masking for extraction
        )
        
        extraction_dataloader = DataLoader(
            extraction_dataset, 
            batch_size=args.batch_size
        )
        
        # Extract embeddings
        embeddings = extract_embeddings(model, extraction_dataloader, device)
        
        # Save embeddings
        if output_dir:
            embeddings_file = os.path.join(output_dir, "embeddings_after_training.npy")
            np.save(embeddings_file, embeddings)
            logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Cluster embeddings
        if args.cluster:
            logger.info(f"Clustering embeddings into {args.n_clusters} clusters")
            clusters, centers, reduced_embeddings = cluster_embeddings(
                embeddings, 
                n_clusters=args.n_clusters
            )
            
            # Save cluster assignments
            if output_dir:
                clusters_file = os.path.join(output_dir, "clusters_after_training.npy")
                np.save(clusters_file, clusters)
                logger.info(f"Saved cluster assignments to {clusters_file}")
                
                # Visualize clusters
                viz_file = os.path.join(output_dir, "clusters_visualization_after_training.png")
                visualize_clusters(reduced_embeddings, clusters, viz_file)
                logger.info(f"Saved cluster visualization to {viz_file}")
    
    # Finish the wandb run
    if not args.disable_wandb:
        if args.val_data and 'best_val_loss' in locals():
            # Create a summary of the training
            wandb.run.summary["best_val_loss"] = best_val_loss
            
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Learning with ESM-2 and LoRA with Weights & Biases tracking")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", 
                        help="ESM-2 model name or path")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="Rank of LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability for LoRA layers")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (FASTA file)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data (FASTA file)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Probability of masking tokens for MLM")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./esm2_lora_unsupervised",
                        help="Directory to save the model and results")
    
    # Mode arguments
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract embeddings, don't train")
    parser.add_argument("--extract_after_training", action="store_true",
                        help="Extract embeddings after training")
    parser.add_argument("--cluster", action="store_true",
                        help="Cluster embeddings")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of clusters")
    
    # Wandb arguments
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="esm2-lora-unsupervised",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")
    
    args = parser.parse_args()
    main(args)