import os
import time
import shutil
from typing import Optional

import fire

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login

from code4chart.init_functions import logger_setup


def main(
        hf_id: Optional[str] = None,
        hf_login_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        verbose: bool = False,
) -> None:
    """
    Download Hugging Face models and tokenizers to the local cache directory.
    "${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"

    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "openai-community/gpt2"
    :param hf_login_token: The login token of the Hugging Face.
    :param cache_dir: The root directory of the cache, e.g., "${HOME}/.cache/huggingface/"
    :param trust_remote_code: Whether to allow for datasets defined on the Hub using a dataset script.
    :param verbose: Verbose mode: show logs.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Logger setup
    logger = logger_setup("DOWNLOAD_HF_MODEL")

    # Parameter checking
    if not isinstance(hf_id, str):
        raise ValueError(f"ValueError: Please specify --hf_id")
    if (cache_dir is None) or (not os.path.isdir(cache_dir)):
        raise ValueError(f"ValueError: Please specify a valid --cache_dir")

    # Hugging Face login
    if isinstance(hf_login_token, str):
        login(token=hf_login_token)

    # Cache directory setup
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    hf_name = "--".join(hf_id.split("/"))
    model_pa_dir = os.path.join(
        cache_dir, "models--" + hf_name, "snapshots")
    model_path = os.path.join(model_pa_dir, "model")
    logger.info(f">>> hf_id = {hf_id}; model_path = {model_path}")
    if os.path.isdir(model_path):
        logger.info(f">>> [START] os.path.isdir(model_path) is True")
        hf_id_local = model_path
    else:
        logger.info(f">>> [START] os.path.isdir(model_path) is False")
        hf_id_local = hf_id

    try:
        # Download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id_local,
            padding_side="left", truncation_side="left",  # "right" for training, "left" for generating
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"len(tokenizer.vocab) = {len(tokenizer.vocab)}")
        logger.info(f"tokenizer.SPECIAL_TOKENS_ATTRIBUTES = {tokenizer.SPECIAL_TOKENS_ATTRIBUTES}")
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    except Exception as e:
        if verbose:
            logger.info(e)

    try:
        # Download the model
        if "Vision" in hf_id or "Qwen2-VL" in hf_id or "llava" in hf_id:
            logger.info(f"Download AutoModelForImageTextToText VLMs (with the processor): {hf_id_local}")
            processor = AutoProcessor.from_pretrained(
                hf_id_local,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id_local,
                torch_dtype=torch.float16,  # torch.bfloat16
                device_map="auto",  # !pip install accelerate
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        else:
            logger.info(f"Download AutoModelForCausalLM Models: {hf_id_local}")
            model = AutoModelForCausalLM.from_pretrained(
                hf_id_local,
                torch_dtype=torch.float16,  # torch.bfloat16
                device_map="auto",  # !pip install accelerate
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of total parameters: {total_params}")
        logger.info(f"Number of trainable parameters: {train_params}")
    except Exception as e:
        if verbose:
            logger.info(e)

    # After downloading, cache_dir/models--xxx/snapshots/yyy/ is the model/tokenizer for from_pretrained(...)
    # Now, we rename cache_dir/hub/models--xxx/snapshots/yyy/ as cache_dir/hub/models--xxx/snapshots/model/
    if os.path.isdir(model_path):
        logger.info(f">>> [END] os.path.isdir(model_path) is True: {model_path}")
    else:
        logger.info(f">>> [END] os.path.isdir(model_path) is False --> Rename")
        fn_list = os.listdir(model_pa_dir)
        logger.info(f">>> os.listdir({model_pa_dir}) = {fn_list}")
        fn_list = [fn for fn in fn_list if not fn.startswith(".")]
        logger.info(f">>> os.listdir({model_pa_dir}) = {fn_list}")
        assert "model" not in fn_list and len(fn_list) == 1
        old_dir = os.path.join(model_pa_dir, fn_list[0])
        shutil.move(old_dir, model_path)
        assert os.path.isdir(model_path)
        logger.info(f">>> [END] The final model_path: {model_path}")
    # To load the local model, set the `model_path` for from_pretrained(...) as follows:
    #   model_path = os.path.join(cache_dir, "models--" + "--".join(hf_id.split("/")), "snapshots/model")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
