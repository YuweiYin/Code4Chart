#!/bin/bash

PARAM=$1

IFS=";"  # Set "," as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"

#echo ${#PARAM_ARRAY[@]}
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  # echo -e ">>> idx = ${idx}; val = ${val}"
  if [[ "${idx}" -eq 1 ]]; then
    TASK=${val}
  fi
  if [[ "${idx}" -eq 2 ]]; then
    BSZ=${val}
  fi
  if [[ "${idx}" -eq 3 ]]; then
    MAX_SEQ_LEN=${val}
  fi
  if [[ "${idx}" -eq 4 ]]; then
    TEXT_LLM=${val}
  fi
  if [[ "${idx}" -eq 5 ]]; then
    CODE_LLM=${val}
  fi
  if [[ "${idx}" -eq 6 ]]; then
    VLM=${val}
  fi
done

if [[ -z ${TASK} ]]
then
  TASK="1"
fi

if [[ -z ${BSZ} ]]
then
  BSZ="1"
fi

if [[ -z ${MAX_SEQ_LEN} ]]
then
  MAX_SEQ_LEN="1024"
fi

if [[ -z ${TEXT_LLM} ]]
then
  TEXT_LLM="meta-llama/Llama-3.1-8B-Instruct"
fi

if [[ -z ${CODE_LLM} ]]
then
  CODE_LLM="meta-llama/CodeLlama-7b-Instruct-hf"
fi

if [[ -z ${VLM} ]]
then
  VLM="meta-llama/Llama-3.2-11B-Vision-Instruct"
fi

#CUDA="0,1"
SEED=42
ROOT_DIR="${HOME}/projects/def-carenini/yuweiyin"
CACHE_DIR="${ROOT_DIR}/.cache/huggingface/"

echo -e "TASK: ${TASK}"
echo -e "BSZ: ${BSZ}"
echo -e "MAX_SEQ_LEN: ${MAX_SEQ_LEN}"
echo -e "TEXT_LLM: ${TEXT_LLM}"
echo -e "CODE_LLM: ${CODE_LLM}"
echo -e "VLM: ${VLM}"
echo -e "ROOT_DIR: ${ROOT_DIR}"
echo -e "CACHE_DIR: ${CACHE_DIR}"

echo -e "\n >>> python3 run_dataset.py --task ${TASK}"
python3 run_dataset.py \
  --task "${TASK}" \
  --cache_dir "${CACHE_DIR}" \
  --project_root_dir "${ROOT_DIR}" \
  --seed ${SEED} \
  --bsz "${BSZ}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --hf_id_text_llm "${TEXT_LLM}" \
  --hf_id_code_llm "${CODE_LLM}" \
  --hf_id_vlm "${VLM}" \
  --verbose

# --interactive
# --show_generation
# --debug
