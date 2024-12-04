#!/bin/bash

echo -e ">>> Yuwei Yin - Apply for Compute Canada Interactive Env (Cedar) <<<\n"

salloc --account=def-carenini \
  --mail-user=yuwei_yin@outlook.com \
  --mail-type=ALL \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --time=0-03:00 --mem=100G --gpus-per-node=v100l:2

#salloc --account=def-carenini --nodes 1 --gres=gpu:v100l:4 --tasks-per-node=4 --mem=16G --time=03:00:00
#salloc --cpus-per-task=4 --gpus-per-node=v100l:1 --mem=32000M --time=01:00:00

# GPU
#echo -e "\n>>> nvidia-smi\n"
#nvidia-smi
