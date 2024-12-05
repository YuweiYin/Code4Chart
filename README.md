# Code4Chart

* Does The Plotting Code Improve Chart Understanding for Vision-Language Models?
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
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"
python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/CodeLlama-7b-Instruct-hf"
python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/CodeLlama-7b-Python-hf"

python3 download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.2-11B-Vision-Instruct"
```

## Dataset Construction

```bash
# Please make sure the "ROOT_DIR" and "CACHE_DIR" variables are correct paths
bash run_dataset.sh "1"
bash run_dataset.sh "2"
bash run_dataset.sh "3"
bash run_dataset.sh "4"
bash run_dataset.sh "5"
```

## Run Experiments

```bash
# Baseline (0, 0, 0, 0, 0): [#item = 63] Accuracy: 0.04762
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=47
python3 run_experiment.py --verbose --task 1 \
  --cache_dir "${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/" --project_root_dir "${HOME}"

# Baseline + Code Input (1, 0, 0, 0, 0): [#item = 63] Accuracy: 0.17460
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=39
python3 run_experiment.py --verbose --task 1 --add_code \
  --cache_dir "${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/" --project_root_dir "${HOME}"

# Baseline + Dataset Info (0, 1, 0, 0, 0): [#item = 63] Accuracy: 0.04762
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=47
python3 run_experiment.py --verbose --task 1 --add_ds_info \
  --cache_dir "${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/" --project_root_dir "${HOME}"

# Baseline + Dataset Info + Code Input (1, 1, 0, 0, 0): [#item = 63] Accuracy: 0.30159
# Done All. Statistics: done_cnt_all=63, miss_cnt_all=53, fail_to_answer_cnt_all=34
python3 run_experiment.py --verbose --task 1 --add_ds_info --add_code \
  --cache_dir "${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/" --project_root_dir "${HOME}"
```

---
