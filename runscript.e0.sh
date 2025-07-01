#!/bin/bash
#SBATCH --job-name=difusco.e1
#SBATCH --output=/home/mananaga/projects/logs/difusco/%j/difusco.out
#SBATCH --error=/home/mananaga/projects/logs/difusco/%j/difusco.out
#SBATCH --time=72:00:00
#SBATCH --partition=preempt
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --hint=nomultithread

# Activate your Conda environment
source ~/.bashrc
conda activate difusco_2_fixed

# Set working directories
DIFUSCO_DIR="/home/mananaga/projects/combinatorial-optimisation/e0"

# Set node count for consistency with generate_tsp_end_to_end.sh
NUM_NODES=50
DATA_DIR="/home/mananaga/projects/data/difusco/TSP/tsp${NUM_NODES}"
CONVERTED_DATA_DIR="${DATA_DIR}/converted_frontierco"
TRAIN_DATASET="${CONVERTED_DATA_DIR}/tsp${NUM_NODES}_frontierco_train.txt"
VALID_DATASET="${CONVERTED_DATA_DIR}/tsp${NUM_NODES}_frontierco_valid.txt"
TEST_DATASET="${CONVERTED_DATA_DIR}/tsp${NUM_NODES}_frontierco_test.txt"

# Setup environment
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set Weights & Biases configuration
export WANDB_ENTITY="mananaga-carnegie-mellon-university"
# Generate wandb run ID
# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

# Change to project directory
cd ${DIFUSCO_DIR}

# Check if datasets exist
if [ ! -f "${TRAIN_DATASET}" ] || [ ! -f "${VALID_DATASET}" ] || [ ! -f "${TEST_DATASET}" ]; then
    echo "Error: Required datasets not found in ${CONVERTED_DATA_DIR}"
    echo "Please run generate_tsp_end_to_end.sh first to create the datasets"
    exit 1
else
    echo "Using datasets from ${CONVERTED_DATA_DIR}"
    echo "Training: $(wc -l < ${TRAIN_DATASET}) instances"
    echo "Validation: $(wc -l < ${VALID_DATASET}) instances"
    echo "Test: $(wc -l < ${TEST_DATASET}) instances"
fi

# Run training with the converted data
echo "Starting DIFUSCO training..."
python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp${NUM_NODES}_frontierco" \
  --wandb_entity "mananaga-carnegie-mellon-university" \
  --diffusion_type "categorical" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/home/mananaga/projects/data" \
  --training_split "${TRAIN_DATASET}" \
  --validation_split "${VALID_DATASET}" \
  --test_split "${TEST_DATASET}" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50
