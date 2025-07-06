#!/bin/bash
# This script runs the ESM-2 LoRA unsupervised learning script with common configurations

# Check if the first argument is provided (path to FASTA file)
if [ -z "$1" ]; then
    echo "Please provide the path to the training FASTA file as the first argument."
    echo "Usage: ./run_lora_esm2.sh <train_fasta> [val_fasta]"
    exit 1
fi

TRAIN_FASTA=$1
VAL_FASTA=$2
OUTPUT_DIR="./esm2_lora_results"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Set model size based on available GPU memory
# You can comment/uncomment depending on your hardware
MODEL_SIZE="small"  # Options: small, medium, large

case $MODEL_SIZE in
    small)
        MODEL_NAME="facebook/esm2_t12_35M_UR50D"
        BATCH_SIZE=8
        MAX_LENGTH=512
        ;;
    medium)
        MODEL_NAME="facebook/esm2_t33_650M_UR50D"
        BATCH_SIZE=4
        MAX_LENGTH=512
        ;;
    large)
        MODEL_NAME="facebook/esm2_t36_3B_UR50D"
        BATCH_SIZE=2
        MAX_LENGTH=512
        ;;
    *)
        echo "Invalid model size: $MODEL_SIZE"
        exit 1
        ;;
esac

# Common parameters
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LEARNING_RATE=5e-5
NUM_EPOCHS=100
SEED=42
N_CLUSTERS=10

echo "Running ESM-2 LoRA training with the following configuration:"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Training data: $TRAIN_FASTA"
echo "Validation data: $VAL_FASTA"
echo "Output directory: $OUTPUT_DIR"

# Command for training and extracting embeddings afterwards
if [ -z "$VAL_FASTA" ]; then
    # Training without validation data
    python finetuneESMLoRA.py \
        --model_name $MODEL_NAME \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --train_data $TRAIN_FASTA \
        --max_length $MAX_LENGTH \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --extract_after_training \
        --cluster \
        --n_clusters $N_CLUSTERS
else
    # Training with validation data
    python finetuneESMLoRA.py \
        --model_name $MODEL_NAME \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --train_data $TRAIN_FASTA \
        --val_data $VAL_FASTA \
        --max_length $MAX_LENGTH \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --extract_after_training \
        --cluster \
        --n_clusters $N_CLUSTERS
fi
