#!/bin/bash
#SBATCH --job-name=train_yolov11_cherryCO
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --error=error_train.err
#SBATCH --output=output_train.out
#SBATCH --nodelist=v100
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabian.escobar@pregrado.uoh.cl


CONTAINER = singularity/yolo11.sif

singularity exec --nv \
    $CONTAINER \
    python3 train.py

