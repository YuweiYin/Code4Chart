# Code4Chart

* C4C: Does Visualization Code Improve Chart Understanding for Vision-Language Models?
* <del>AutoDA: Automated Data Analysis via Multimodal Large Language Models</del>
* <del>Code, Chart, and Caption: A Synthetic Dataset for Chart Understanding and Generation</del>

---

## Python Environment

```bash
#python3 -m venv c4c
conda create -n c4c -y python=3.10
conda activate c4c

pip3 install -r requirements.txt -i https://pypi.org/simple/
pip3 install -e . -i https://pypi.org/simple/
```

### Download Models

```bash
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"  # YOUR CACHE_DIR
python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/CodeLlama-7b-Instruct-hf"
#python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
#  --hf_id "meta-llama/CodeLlama-7b-Python-hf"

python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.2-11B-Vision-Instruct"  # The VLM to generate chart captions (and for evaluation)
python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "Qwen/Qwen2-VL-7B-Instruct"  # The VLM for evaluation
```

## Dataset Construction

```bash
# Please make sure the "ROOT_DIR" and "CACHE_DIR" variables are correct paths
bash run_dataset.sh "1"  # step1_get_metadata()
bash run_dataset.sh "2"  # step2_analyze_da_reqs() <-- Text LLM (GPU needed)
bash run_dataset.sh "3"  # step3_gen_vis_code() <-- Code LLM (GPU needed)
bash run_dataset.sh "4"  # step4_vis_code_postprocess()
bash run_dataset.sh "5"  # step5_exec_vis_code()
bash run_dataset.sh "6"  # step6_chart_cap_gen() <-- VLM (GPU needed)
bash run_dataset.sh "7"  # step7_overall_analysis() <-- Text LLM (GPU needed)
bash run_dataset.sh "8"  # step8_merge_all_info()
bash run_dataset.sh "9"  # step9_chart_qa_task() <-- Text LLM (GPU needed)
```

## Run Experiments

```bash
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"  # YOUR CACHE_DIR
HF_ID_VLM="meta-llama/Llama-3.2-11B-Vision-Instruct"  # The VLMs to evaluate
#HF_ID_VLM="Qwen/Qwen2-VL-7B-Instruct"

# Baseline (0, 0, 0, 0, 0): [#item = 63] Accuracy: 0.04762
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=47
python3 run_experiment.py --verbose --task 1 \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"

# Baseline + Code Input (1, 0, 0, 0, 0): [#item = 63] Accuracy: 0.17460
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=39
python3 run_experiment.py --verbose --task 1 --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"

# Baseline + Dataset Info (0, 1, 0, 0, 0): [#item = 63] Accuracy: 0.04762
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=47
python3 run_experiment.py --verbose --task 1 --add_ds_info \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"

# Baseline + Dataset Info + Code Input (1, 1, 0, 0, 0): [#item = 63] Accuracy: 0.30159
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=34
python3 run_experiment.py --verbose --task 1 --add_ds_info --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
```

---
