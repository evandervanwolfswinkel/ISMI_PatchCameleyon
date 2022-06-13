#!/bin/bash
#SBATCH --job-name=evander_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time 48:00:00
#SBATCH --error /home/jkoopmans/logs/%A_%a.err
#SBATCH --output /home/jkoopmans/logs/%A_%a.out


RUN_NAME="densenet121_augmented_nofreeze_epoch25_nopretrained"

BACKBONE_NAME="densenet121" # should match the name in helper_funcs.py -> get_backbone()
PRETRAINED=false
EXTRA_AUGMENTATIONS=true
FREEZE=false
EPOCHS=25

DATA_DIR="/scratch-shared/jkoopmans/data/googlefolder"
SUBMISSIONS_FOLDER="${HOME}/evander/submissions"
CHECKPOINTS_FOLDER="${HOME}/evander/checkpoints/"
TTA=false


# Run the train script
/home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/main.py \
$DATA_DIR \
${SLURM_JOB_ID} \
$SUBMISSIONS_FOLDER \
$CHECKPOINTS_FOLDER \
$BACKBONE_NAME \
$EXTRA_AUGMENTATIONS \
$FREEZE \
$PRETRAINED \
$EPOCHS \
$RUN_NAME

# Get the paths to the best and last checkpoint
BEST_CHECKPOINT_PATH=$(readlink -f $CHECKPOINTS_FOLDER/${SLURM_JOB_ID}_best_*)
LAST_CHECKPOINT_PATH=$(readlink -f $CHECKPOINTS_FOLDER/${SLURM_JOB_ID}_last_*)

# Create a submission file for the best checkpoint
/home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/test.py \
$DATA_DIR \
$BEST_CHECKPOINT_PATH \
$SUBMISSIONS_FOLDER/$(basename $BEST_CHECKPOINT_PATH .pth.tar).csv \
$BACKBONE_NAME \
$TTA

# Create a submission file for the last checkpoint
/home/jkoopmans/miniconda3/envs/torch/bin/python /home/jkoopmans/pcam/test.py \
$DATA_DIR \
$LAST_CHECKPOINT_PATH \
$SUBMISSIONS_FOLDER/$(basename $LAST_CHECKPOINT_PATH .pth.tar).csv \
$BACKBONE_NAME \
$TTA