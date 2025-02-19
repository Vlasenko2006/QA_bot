#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name= set your job name
#SBATCH --nodes= set number of nodes  
#SBATCH --ntasks-per-node= set numper of tasks per node
#SBATCH --cpus-per-task= set number of cpu per tasks
#SBATCH --time= set time limit if needed, or remove this line
#SBATCH --account= set your account if needed or remove this line
#SBATCH --partition= specify your partition
#SBATCH --error= file for error messages
#SBATCH -o name of the output file

#SBATCH --exclusive                
#SBATCH --mem=0                    



srun $HOME/gpt2_finetuning_env/bin/python3.10 QA_bot.py # specify the path to yoyr python environment. Here is the default one
