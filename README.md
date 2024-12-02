# Code4Chart

* Does The Plotting Code Improve Chart Understanding for Vision-Language Models?
* <del>AutoDA: Automated Data Analysis via Multimodal Large Language Models</del>
* <del>Code, Chart, and Caption: A Synthetic Dataset for Chart Understanding and Generation</del>

---

## Python Environment

```bash
pip3 install -r requirements.txt -i https://pypi.org/simple/
pip3 install -e . -i https://pypi.org/simple/
```

### Download Models

```bash
CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"
python3 utils/download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 utils/download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/CodeLlama-7b-Instruct-hf"
python3 utils/download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/CodeLlama-7b-Python-hf"

python3 utils/download_hf_model.py --trust_remote_code --verbose --cache_dir "${CACHE_DIR}" \
  --hf_id "meta-llama/Llama-3.2-11B-Vision-Instruct"
```

---
