#!/bin/bash
#SBATCH -J qwen_omni_finetune_fast             # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 720                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem=80G
#SBATCH -q coc-ice
#SBATCH -o./logs/qwen_omni_fast/Report-%j.out        # Combined output and error messages
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail preferences
#SBATCH --mail-user=shubohan@gatech.edu  # E-mail address for notificationss

# Load conda environment
cd /home/hice1/bshu30/MGT6785
module load anaconda3
conda activate mgt6785

# Print job info
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "================================"

# Change to working directory
cd /home/hice1/bshu30/MGT6785

# Run training
python train_qwen_omni_fast.py

# Print completion
echo "================================"
echo "End time: $(date)"
echo "================================"
