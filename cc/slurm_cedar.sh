#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --mail-user=yuwei_yin@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=v100l:2
#SBATCH --mem=100G
#SBATCH --time=0-12:00  # time (DD-HH:MM)

#VENV=$1  # venv name
SCRIPT=$1  # bash script name

#bash cc_server_info.sh
#bash cc_server_init.sh "${VENV}"
bash "${SCRIPT}"

#SBATCH --gpus-per-node=p100:2
#SBATCH --gpus-per-node=p100l:2
#SBATCH --gpus-per-node=v100l:2

#SBATCH --exclusive
#SBATCH --job-name=job_name
#SBATCH --output=output.txt
#SBATCH --error=error.txt
