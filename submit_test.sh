#!/bin/bash
#SBATCH --job-name=evander_train
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time 10:00:00
#SBATCH --error /home/jkoopmans/logs/%A_%a.err
#SBATCH --output /home/jkoopmans/logs/%A_%a.out


RUN_NAME="_best_efficientnet_b0_25epochs_NoPretrain_NoFreeze_LimitedAugs.pth"
BACKBONE_NAME="efficientnet_b0" # should match the name in helper_funcs.py -> get_backbone()
TTA=true


DATA_DIR="/scratch-shared/jkoopmans/data/googlefolder"
SUBMISSIONS_FOLDER="${HOME}/evander/submissions"
CHECKPOINTS_FOLDER="${HOME}/evander/checkpoints/"
SLURM_JOB=1109229


# Get the paths to the best and last checkpoint
BEST_CHECKPOINT_PATH=$(readlink -f $CHECKPOINTS_FOLDER/${SLURM_JOB}_best_*)
#LAST_CHECKPOINT_PATH=$(readlink -f $CHECKPOINTS_FOLDER/${SLURM_JOB}_last_*)

for i in {1..5}
do
# Create a submission file for the best checkpoint
    /home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/test.py \
    $DATA_DIR \
    $BEST_CHECKPOINT_PATH \
    $SUBMISSIONS_FOLDER/$(basename $BEST_CHECKPOINT_PATH .pth.tar)_TTA${i}_.csv \
    $BACKBONE_NAME \
    $TTA
done


# Create a submission file for the best checkpoint
# /home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/test.py \
# $DATA_DIR \
# $BEST_CHECKPOINT_PATH \
# $SUBMISSIONS_FOLDER/$(basename $BEST_CHECKPOINT_PATH .pth.tar)_noTTA.csv \
# $BACKBONE_NAME \
# $TTA



# Create a submission file for the last checkpoint
# /home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/test.py \
# $DATA_DIR \
# $LAST_CHECKPOINT_PATH \
# $SUBMISSIONS_FOLDER/$(basename $LAST_CHECKPOINT_PATH .pth.tar).csv \
# $BACKBONE_NAME \
# $TTA