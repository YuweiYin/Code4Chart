#!/bin/bash

PARAM=$1

IFS=","  # Set "," as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"

#echo ${#PARAM_ARRAY[@]}
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  # echo -e ">>> idx = ${idx}; val = ${val}"
  if [[ "${idx}" -eq 1 ]]; then
    BSZ=${val}
  fi
  if [[ "${idx}" -eq 2 ]]; then
    MAX_SEQ_LEN=${val}
  fi
  if [[ "${idx}" -eq 3 ]]; then
    TEXT_LLM=${val}
  fi
  if [[ "${idx}" -eq 4 ]]; then
    CODE_LLM=${val}
  fi
  if [[ "${idx}" -eq 5 ]]; then
    VLM=${val}
  fi
done

if [[ -z ${BSZ} ]]
then
  BSZ="1"
fi

if [[ -z ${MAX_SEQ_LEN} ]]
then
  MAX_SEQ_LEN="512"
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

echo -e "TEXT_LLM: ${TEXT_LLM}"
echo -e "CODE_LLM: ${CODE_LLM}"
echo -e "VLM: ${VLM}"
echo -e "BSZ: ${BSZ}"
echo -e "MAX_SEQ_LEN: ${MAX_SEQ_LEN}"

#CUDA="0,1"
SEED=42
ROOT_DIR="${HOME}/projects/def-carenini/yuweiyin"
CACHE_DIR="${ROOT_DIR}/.cache/huggingface/"

echo -e "\n >>> python3 run_dataset.py --task 1"
python3 run_dataset.py \
  --task 1 \
  --cache_dir "${CACHE_DIR}" \
  --project_root_dir "${ROOT_DIR}" \
  --seed ${SEED} \
  --hf_id_text_llm "${TEXT_LLM}" \
  --hf_id_code_llm "${CODE_LLM}" \
  --hf_id_vlm "${VLM}" \
  --bsz "${BSZ}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --verbose

# --interactive
# --show_generation
# --debug
