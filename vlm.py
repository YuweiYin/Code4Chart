import os
# import sys
# import json
# import time
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class VLM:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            cache_dir: Optional[str] = None,
            data_dir: str = os.path.join(os.path.expanduser("~"), "data/auto_da/"),
            project_root_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
            bsz: int = 1,
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Text-to-Text LLM

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param cache_dir: The root directory of the cache.
        :param data_dir: The directory of the data root.
        :param project_root_dir: The directory of the project root.
        :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "listen2you002/ChartLlama-13b"
        :param bsz: The batch size.
        :param show_generation: Whether to show outputs during generation.
        :param debug: Debugging / developing mode.
        :return: None.
        """

        # https://llava-vl.github.io/   "liuhaotian/llava-v1.5-7b"
        # https://github.com/tingxueronghua/ChartLlama-code   "listen2you002/ChartLlama-13b"
        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.data_dir = cache_dir
        self.data_dir = data_dir
        self.project_root_dir = project_root_dir
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/auto_da")
        self.home_dir = os.path.expanduser("~")
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.bsz = bsz
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug

        # Data directory
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Cache directory
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.project_root_dir, ".cache/huggingface/")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)

        if self.verbose:
            self.logger.info(f">>> home_dir: {self.home_dir}")
            self.logger.info(f">>> project_root_dir: {self.project_root_dir}")
            self.logger.info(f">>> cache_dir: {self.cache_dir}")
            self.logger.info(f">>> data_dir: {self.data_dir}")
            self.logger.info(f">>> ckpt_dir: {self.ckpt_dir}")

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        self.model_path = os.path.join(
            self.cache_dir, "models--" + self.hf_name, "snapshots/model")
        assert os.path.isdir(self.model_path), f"AssertionError: assert os.path.isdir({self.model_path})"

        # Load tokenizer
        tokenizer_gen = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left", truncation_side="left",  # "right" for training, "left" for generating
            cache_dir=self.cache_dir,
        )
        # tokenizer_gen.add_special_tokens({"pad_token": "<|pad_of_text|>"})
        tokenizer_gen.pad_token = tokenizer_gen.eos_token
        tokenizer_gen.pad_token_id = tokenizer_gen.eos_token_id

        self.terminators_gen = [
            tokenizer_gen.eos_token_id,
            # tokenizer_gen.convert_tokens_to_ids("<|eot_id|>")
            tokenizer_gen.convert_tokens_to_ids(tokenizer_gen.eos_token)
        ]

        self.tokenizer_gen = tokenizer_gen
        # self.model = None
        self.model = self.load_model()

    def load_model(
            self,
    ):
        # Load the model
        cur_model_path = self.hf_id
        model = AutoModelForCausalLM.from_pretrained(
            cur_model_path,
            torch_dtype=torch.float16,  # torch.bfloat16
            # torch_dtype=torch.float8_e5m2,  # torch.float8
            device_map="auto",  # !pip install accelerate
            # device_map=self.cuda_dict["device"] if self.debug else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # model = model.to(device=self.cuda_dict["device"])
        # list(model.state_dict().keys())
        model.generation_config.pad_token_id = self.tokenizer_gen.pad_token_id  # eos_token_id
        # model.resize_token_embeddings(len(self.tokenizer_train))  # if added new special tokens (Option 1)
        # model.train()

        model.eval()
        # Set all modules as non-trainable
        trainable_param_names = []
        for p_name, param in model.named_parameters():
            if any(tpn in p_name for tpn in trainable_param_names):
                param.requires_grad = True
            else:
                param.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f"Number of total parameters ({cur_model_path}): {total_params}")
            self.logger.info(f"Number of trainable parameters (cur_model_path): {train_params}")

        return model

    def run_generation(
            self,
    ):
        # TODO: Input prompts (Chart Caption; Chart Insight)
        # Load the tabular data and data analysis requirements/instructions
        inputs = None

        return None
