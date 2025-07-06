# ESM2 Mutation Effect Prediction
# 
# This notebook uses your fine-tuned ESM2 model to predict the effects of
# single amino acid mutations in a protein sequence.

import torch
from transformers import EsmForMaskedLM, EsmTokenizer
import re
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Function to parse mutation strings like 'A45G'
def parse_mutation(mutation_str):
    """
    Parse a mutation string into wild type, position, and mutant amino acid.
    
    Example: 'A45G' -> ('A', 45, 'G')
    """
    pattern = r"([A-Z])(\d+)([A-Z])"
    match = re.match(pattern, mutation_str)
    if not match:
        raise ValueError(f"Invalid mutation format: {mutation_str}. Expected format: 'A45G'")
    wt, pos, mut = match.groups()
    return wt, int(pos), mut

# Function to introduce a mutation into a sequence
def introduce_mutation(sequence, position, mutation):
    """
    Create a new sequence with a single amino acid substitution.
    
    Args:
        sequence: Original protein sequence
        position: Position to mutate (1-indexed)
        mutation: New amino acid to insert
        
    Returns:
        New sequence with the mutation
    """
    pos_idx = position - 1  # Convert to 0-based indexing
    
    # Check position is valid
    if pos_idx >= len(sequence):
        raise ValueError(f"Position {position} is out of bounds for sequence of length {len(sequence)}")
    
    # Create and return the mutated sequence
    return sequence[:pos_idx] + mutation + sequence[pos_idx + 1:]

# Load the model and tokenizer
# Modified load_model function that works with fine-tuned checkpoints
def load_model(model_path, base_model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Load the fine-tuned ESM2 model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model directory
        base_model_name: Name of the base ESM2 model that was fine-tuned
        
    Returns:
        Loaded model, tokenizer, and device
    """
    print(f"Loading model from {model_path}")
    print(f"Using base model tokenizer from {base_model_name}")
    
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the tokenizer from the base model, not the fine-tuned checkpoint
    tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    
    # Load the fine-tuned model
    model = EsmForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, tokenizer, device

# Function to compute mutation score
def compute_mutation_score(model, tokenizer, wild_type_seq, mutated_seq, device):
    """
    Calculate a score for a mutation by comparing model predictions.
    
    This function:
    1. Runs both the wild-type and mutated sequences through the model
    2. Computes log probabilities for each sequence
    3. Determines how much more/less likely the mutated sequence is according to the model
    
    The score can be interpreted as:
    - Positive: Mutation may be beneficial
    - Near zero: Mutation is likely neutral
    - Negative: Mutation may be detrimental
    
    Args:
        model: The ESM2 model
        tokenizer: The ESM2 tokenizer
        wild_type_seq: Original protein sequence
        mutated_seq: Protein sequence with mutation
        device: Computation device (CPU/GPU)
        
    Returns:
        Score representing predicted effect of mutation
    """
    # Tokenize both sequences
    wt_tokens = tokenizer(wild_type_seq, return_tensors="pt").to(device)
    mut_tokens = tokenizer(mutated_seq, return_tensors="pt").to(device)
    
    # Get model outputs
    with torch.no_grad():
        wt_output = model(**wt_tokens)
        mut_output = model(**mut_tokens)
        print(f"mut_output is: \n {mut_output}")
    
    # Get log probabilities
    wt_logits = wt_output.logits.squeeze()
    mut_logits = mut_output.logits.squeeze()
    #print(f"mut_logits are: \n {mut_logits}")
    
    wt_log_probs = torch.nn.functional.log_softmax(wt_logits, dim=-1)
    mut_log_probs = torch.nn.functional.log_softmax(mut_logits, dim=-1)
    
    # Calculate average difference in sequence log probability
    seq_log_prob_diff = (mut_log_probs.sum() - wt_log_probs.sum()) / len(wild_type_seq)
    #print("mutated probs are: \n {mut_log_probs}")
    return seq_log_prob_diff.item()

# Function to predict multiple mutations
def predict_mutations(model, tokenizer, device, wild_type_seq, mutations):
    """
    Predict the impact of multiple mutations on a wild-type sequence.
    
    Args:
        model: The ESM2 model
        tokenizer: The ESM2 tokenizer
        device: Computation device
        wild_type_seq: The reference protein sequence
        mutations: List of mutation strings (e.g., ['A45G', 'D134E'])
    
    Returns:
        DataFrame with mutation information and scores
    """
    results = []
    
    # Process each mutation
    for mut_str in tqdm(mutations, desc="Processing mutations"):
        try:
            # Parse the mutation
            wt, pos, mut = parse_mutation(mut_str)
            
            # Verify wild-type matches the sequence
            if wild_type_seq[pos-1] != wt:
                print(f"Warning: Expected {wt} at position {pos}, but found {wild_type_seq[pos-1]}")
                continue
                
            # Create the mutated sequence
            mutated_seq = introduce_mutation(wild_type_seq, pos, mut)
            
            # Compute the mutation score
            score = compute_mutation_score(model, tokenizer, wild_type_seq, mutated_seq, device)
            
            # Store the result
            results.append({
                "mutation": mut_str,
                "position": pos,
                "wild_type": wt,
                "mutant": mut,
                "predicted_score": score
            })
            
        except Exception as e:
            print(f"Error processing mutation {mut_str}: {str(e)}")
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    
    return results_df

# Function to validate against DMS data
def validate_against_dms(predictions_df, dms_df, correlation_method='spearman'):
    """
    Compare model predictions with experimental DMS data.
    
    Args:
        predictions_df: DataFrame with model predictions
        dms_df: DataFrame with DMS experimental data
        correlation_method: 'spearman' or 'pearson'
        
    Returns:
        Correlation value and merged DataFrame
    """
    # Merge the predictions with DMS data
    merged_df = predictions_df.merge(
        dms_df, 
        left_on="mutation", 
        right_on="mutant", 
        how="inner"
    )
    
    # Calculate correlation
    if correlation_method == 'spearman':
        corr, p_value = stats.spearmanr(merged_df['predicted_score'], merged_df['DMS_score'])
    else:
        corr, p_value = stats.pearsonr(merged_df['predicted_score'], merged_df['DMS_score'])
    
    print(f"{correlation_method.capitalize()} correlation: {corr:.3f} (p-value: {p_value:.6f})")
    print(f"Number of mutations compared: {len(merged_df)}")
    
    return corr, merged_df

# Let's put it all together in a usage example