#!/bin/bash

# Set the Python path to include the current directory
export PYTHONPATH="$PWD:$PYTHONPATH"

# Use a smaller TSP instance for quick testing
TSP_SIZE=50
TRAIN_WEIGHT=1.0   # Constraint weight for training (0.0 to disable)
SAMPLE_WEIGHT=0.5  # Constraint weight for sampling (0.0 to disable)

# Training with tour constraints
python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_tour_constraint_tsp${TSP_SIZE}" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./data" \
  --training_split "./data/tsp${TSP_SIZE}_train_concorde.txt" \
  --validation_split "./data/tsp${TSP_SIZE}_valid_concorde.txt" \
  --test_split "./data/tsp${TSP_SIZE}_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --train_constraint_weight ${TRAIN_WEIGHT} \
  --sample_constraint_weight ${SAMPLE_WEIGHT}

# To test different configurations:
# 1. Training-only constraints: --train_constraint_weight 1.0 --sample_constraint_weight 0.0
# 2. Sampling-only constraints: --train_constraint_weight 0.0 --sample_constraint_weight 1.0 
# 3. Both with different weights: --train_constraint_weight 1.0 --sample_constraint_weight 0.5

# To run inference only on a trained model, use:
# --do_test --ckpt_path "/path/to/checkpoint.ckpt" --resume_weight_only 