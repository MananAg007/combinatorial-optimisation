#!/bin/bash
# Script to generate TSP instances, solve them, and convert to DIFUSCO format


set -e  # Exit on error
conda activate difusco

# ==================== Configuration ====================
# Paths
FRONTIERCO_DIR="/home/mananaga/projects/combinatorial-optimisation/FrontierCO"
DIFUSCO_DIR="/home/mananaga/projects/combinatorial-optimisation/difusco"
SCRIPTS_DIR="/home/mananaga/projects/runscripts"

# LKH path (update this with your LKH path)
LKH_PATH="/home/mananaga/projects/combinatorial-optimisation/LKH-3.0.13/LKH"

# Instance generation parameters
MIN_NODES=1000
MAX_NODES=1000
TRAIN_INSTANCES=64000
VALID_INSTANCES=1000
TEST_INSTANCES=1000
SEED=42

# Directory structure
DATA_DIR="${FRONTIERCO_DIR}/training_files/data/TSP"
TRAIN_DIR="${DATA_DIR}/train_instances"
VALID_DIR="${DATA_DIR}/valid_instances"
TEST_DIR="${DATA_DIR}/test_instances"
TRAIN_SOL_DIR="${DATA_DIR}/train_instances_sol"
VALID_SOL_DIR="${DATA_DIR}/valid_instances_sol"
TEST_SOL_DIR="${DATA_DIR}/test_instances_sol"
TRAIN_PAR_DIR="${DATA_DIR}/train_instances_par"
VALID_PAR_DIR="${DATA_DIR}/valid_instances_par"
TEST_PAR_DIR="${DATA_DIR}/test_instances_par"

# DIFUSCO converted data
DIFUSCO_DATA_DIR="${DIFUSCO_DIR}/data/converted_frontierco"
DIFUSCO_TRAIN="${DIFUSCO_DATA_DIR}/tsp1000_frontierco_train.txt"
DIFUSCO_VALID="${DIFUSCO_DATA_DIR}/tsp1000_frontierco_valid.txt"
DIFUSCO_TEST="${DIFUSCO_DATA_DIR}/tsp1000_frontierco_test.txt"

# ==================== Setup ====================
# Create directories
mkdir -p "${TRAIN_DIR}" "${VALID_DIR}" "${TEST_DIR}"
mkdir -p "${TRAIN_SOL_DIR}" "${VALID_SOL_DIR}" "${TEST_SOL_DIR}"
mkdir -p "${DIFUSCO_DATA_DIR}"

# ==================== Generate TSP Instances ====================
echo "Generating training instances..."
python "${DATA_DIR}/generate_training_instances.py" \
  --min_nodes ${MIN_NODES} \
  --max_nodes ${MAX_NODES} \
  --num_instances ${TRAIN_INSTANCES} \
  --seed ${SEED} \
  --output "${TRAIN_DIR}"

echo "Generating validation instances..."
python "${DATA_DIR}/generate_training_instances.py" \
  --min_nodes ${MIN_NODES} \
  --max_nodes ${MAX_NODES} \
  --num_instances ${VALID_INSTANCES} \
  --seed $((SEED+1)) \
  --output "${VALID_DIR}"

echo "Generating test instances..."
python "${DATA_DIR}/generate_training_instances.py" \
  --min_nodes ${MIN_NODES} \
  --max_nodes ${MAX_NODES} \
  --num_instances ${TEST_INSTANCES} \
  --seed $((SEED+2)) \
  --output "${TEST_DIR}"

# ==================== Create LKH Parameter Files ====================
echo "Creating parameter files for training instances..."
bash "${DATA_DIR}/generate_par.sh" "${TRAIN_DIR}"

echo "Creating parameter files for validation instances..."
bash "${DATA_DIR}/generate_par.sh" "${VALID_DIR}"

echo "Creating parameter files for test instances..."
bash "${DATA_DIR}/generate_par.sh" "${TEST_DIR}"

# ==================== Solve TSP Instances ====================
echo "Solving training instances..."
bash "${DATA_DIR}/solve.sh" \
  --instance-dir "${TRAIN_DIR}" \
  --lkh-path "${LKH_PATH}" \
  --parallel 10

echo "Solving validation instances..."
bash "${DATA_DIR}/solve.sh" \
  --instance-dir "${VALID_DIR}" \
  --lkh-path "${LKH_PATH}" \
  --parallel 10

echo "Solving test instances..."
bash "${DATA_DIR}/solve.sh" \
  --instance-dir "${TEST_DIR}" \
  --lkh-path "${LKH_PATH}" \
  --parallel 10

# ==================== Convert to DIFUSCO Format ====================
echo "Converting training instances to DIFUSCO format..."
python "${DIFUSCO_DIR}/convert_tsp_instances.py" \
  --instance_dir "${TRAIN_DIR}" \
  --solution_dir "${TRAIN_SOL_DIR}" \
  --output_dir "${DIFUSCO_DATA_DIR}" \
  --output_file "tsp1000_frontierco_train.txt"

echo "Converting validation instances to DIFUSCO format..."
python "${DIFUSCO_DIR}/convert_tsp_instances.py" \
  --instance_dir "${VALID_DIR}" \
  --solution_dir "${VALID_SOL_DIR}" \
  --output_dir "${DIFUSCO_DATA_DIR}" \
  --output_file "tsp1000_frontierco_valid.txt"

echo "Converting test instances to DIFUSCO format..."
python "${DIFUSCO_DIR}/convert_tsp_instances.py" \
  --instance_dir "${TEST_DIR}" \
  --solution_dir "${TEST_SOL_DIR}" \
  --output_dir "${DIFUSCO_DATA_DIR}" \
  --output_file "tsp1000_frontierco_test.txt"
