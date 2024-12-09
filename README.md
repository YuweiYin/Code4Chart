# Code4Chart (C4C)

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
bash run_dataset.sh "10"  # step10_chart_qa_edit_chart()
```

## Experiments

### Main Experiment 1

**Research Question**: Does the VisCode improve VLMs in chart understanding? **Yes**

- **Settings**: **63** generated chart figures with questions, options, and answers.

```bash
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"  # YOUR CACHE_DIR
HF_ID_VLM="meta-llama/Llama-3.2-11B-Vision-Instruct"  # The VLMs for evaluation
#HF_ID_VLM="Qwen/Qwen2-VL-7B-Instruct"

# Main (w/ or w/o VisCode as input)
python3 run_experiment.py --verbose --task 1 \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
python3 run_experiment.py --verbose --task 1 --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"

# Analytical (with dataset information as extra input)
python3 run_experiment.py --verbose --task 1 --add_ds_info \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
python3 run_experiment.py --verbose --task 1 --add_ds_info --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
```

| Setting           | Accuracy  | 
|-------------------|-----------|
| w/o VisCode       | 49.2%     | 
| w/ VisCode (Ours) | **52.4%** | 

### Main Experiment 2

**Research Question**: Does the VisCode make VLMs more robust to chart modifications? **Yes**

- **Settings**
  - Step 1: **Pick 7 QA examples** that **Acc=100%** using original charts (w/ or w/o VisCode);
  - Step 2: Modify the original VisCode, only change the color of bars (**8 new colors** each);
  - Step 3: Produce new chart figures (paired with the **original QA**);
  - Step 4: Run VLM evaluation on charts with new colors. (7 * 8 = **54 new examples**)

```bash
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"  # YOUR CACHE_DIR
HF_ID_VLM="meta-llama/Llama-3.2-11B-Vision-Instruct"  # The VLMs for evaluation
#HF_ID_VLM="Qwen/Qwen2-VL-7B-Instruct"

# Main (w/ or w/o VisCode as input)
python3 run_experiment_edit.py --verbose --task 1 \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
python3 run_experiment_edit.py --verbose --task 1 --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"

# Analytical (with dataset information as extra input)
python3 run_experiment_edit.py --verbose --task 1 --add_ds_info \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
python3 run_experiment_edit.py --verbose --task 1 --add_ds_info --add_code \
  --cache_dir "${CACHE_DIR}" --project_root_dir "${HOME}" --hf_id_vlm "${HF_ID_VLM}"
```

| Setting           | Accuracy  | Δ Acc  | Avg Output Len |
|-------------------|-----------|--------|----------------|
| w/o VisCode       | 85.7%     | -14.3% | 97.7           |
| w/ VisCode (Ours) | **100%**  | **0**  | **36.6**       |

---
