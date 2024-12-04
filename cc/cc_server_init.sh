#!/bin/bash

VENV=$1
echo -e ">>> Yuwei Yin - Compute Canada Server Init --- ${VENV} <<<\n"

# Cache dir
cache_dir="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"
echo -e "cache_dir=${cache_dir} \n"

# Project directory
echo -e "cd ${HOME}/projects/def-carenini/yuweiyin/projects/Code4Chart/"
echo "module load StdEnv/2023 python/3.10 arrow/14.0.0"
echo ". ${HOME}/projects/def-carenini/yuweiyin/venv/c4c/bin/activate"
echo -e "module load StdEnv/2023 python/3.10 arrow/14.0.0 \n"

#echo "module load StdEnv/2023 python/3.10 arrow/14.0.0"
#module load StdEnv/2023 python/3.10 arrow/14.0.0

echo -e "squeue -u yuweiyin"
echo -e "diskusage_report"
echo -e "scancel -u yuweiyin -t PENDING"


echo "pip3 install -r requirements.txt -i https://pypi.org/simple/"
echo "pip3 install -e . -i https://pypi.org/simple/"

echo "which python3"
echo "python3 -V"
echo "pip3 list | wc -l"
