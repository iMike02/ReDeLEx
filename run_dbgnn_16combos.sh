#!/bin/bash
#SBATCH --job-name=dbgnn-16combos
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#### --array=0-15


# declare -a combinations=(
#     'False default False default_combinations'
#     'False default True default_combinations'
#     'False default True keep_attributes'
#     'False default True keep_table'
#     'True default False default_combinations'
#     'True keep_attributes False default_combinations'
#     'True keep_table False default_combinations'
#     'True default True default_combinations'
#     'True default True keep_attributes'
#     'True default True keep_table'
#     'True keep_attributes True default_combinations'
#     'True keep_attributes True keep_attributes'
#     'True keep_attributes True keep_table'
#     'True keep_table True default_combinations'
#     'True keep_table True keep_attributes'
#     'True keep_table True keep_table'
# )
process_bridge='False'
bridge_strategy='default'
process_hub='False'
hub_strategy='default_combinations'

# -----------------------------TADY VYBÍRÁM DATASET A TASK------------------------------
dataset='rel-stack'
task='user-engagement'
# --------------------------------------------------------------------------------------

VENV_PATH="/home/gabrimi8/RDL/ReDeLEx/.venv"

# Activate the local virtual environment
source "${VENV_PATH}/bin/activate"

EXPERIMENT_NAME="dbgnn_16combos"

echo $SLURM_ARRAY_JOB_ID

# START_TIME=$(sacct -j ${SLURM_JOB_ID} --format=Start -n | head -n 1)

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

# NUM_SAMPLES=5

# MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir



# Run experiment with different params

# combo=${combinations[$SLURM_ARRAY_TASK_ID]}
# # Split into 4 variables
# read -r process_bridge bridge_strategy process_hub hub_strategy <<< "$combo"



combo_id="${SLURM_ARRAY_TASK_ID}"
log_dir=${experiment_dir}/${dataset}_${task}
mkdir -p $log_dir
mkdir -p "${log_dir}/run_logs"

cmd=(
  python -u experiments/original/dbgnn_train_5seedsingle.py
  --dataset="${dataset}"
  --task="${task}"
  --bridge_strategy="${bridge_strategy}"
  --hub_strategy="${hub_strategy}"
  --log_dir="${experiment_dir}/${dataset}_${task}"
)

if [[ "${process_bridge}" == "True" ]]; then
  cmd+=(--process_bridge)
fi

if [[ "${process_hub}" == "True" ]]; then
  cmd+=(--process_hub)
fi

"${cmd[@]}" &> "${log_dir}/run_logs/run${combo_id}.log"