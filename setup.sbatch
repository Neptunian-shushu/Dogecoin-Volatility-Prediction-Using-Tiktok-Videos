#!/bin/bash
SBATCH -J crypto              # Job name
SBATCH -N1 --gres=gpu:H100:1
SBATCH -t 180                    # Duration of the job (Ex: 15 mins)
SBATCH --mem-per-cpu=5G
SBATCH -q coc-ice
