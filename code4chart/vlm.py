import os
# import sys
# import json
# import time
from typing import Optional

import torch
from transformers import AutoTokenizer  # AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText


class VLM:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
            bsz: int = 1,
            max_seq_len: int = 1024,
            show_generation: bool = False,
            debug: bool = False,
            load_model: bool = True,
    ):
        """
        Text-to-Text LLM

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param cache_dir: The root directory of the cache.
        :param project_root_dir: The directory of the project root.
        :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "listen2you002/ChartLlama-13b"
        :param bsz: The batch size.
        :param show_generation: Whether to show outputs during generation.
        :param max_seq_len: The maximum sequence length for padding/truncation.
        :param debug: Debugging / developing mode.
        :param load_model: Whether to load the pre-trained model.
        :return: None.
        """

        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
        # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
        # https://llava-vl.github.io/   "liuhaotian/llava-v1.5-7b"
        # https://github.com/tingxueronghua/ChartLlama-code   "listen2you002/ChartLlama-13b"

        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.cache_dir = cache_dir
        self.project_root_dir = project_root_dir
        self.home_dir = os.path.expanduser("~")
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug

        # Data and checkpoint directory
        self.data_dir = os.path.join(project_root_dir, "data/code4chart")
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/code4chart")
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
        self.tokenizer_gen = self.load_tokenizer(
            model_path=self.model_path, padding_side="left", truncation_side="left")  # "left" for generating
        self.terminators_gen = [
            self.tokenizer_gen.eos_token_id,
            # self.tokenizer_gen.convert_tokens_to_ids("<|eot_id|>")
            self.tokenizer_gen.convert_tokens_to_ids(self.tokenizer_gen.eos_token)
        ]

        # Load the model
        if load_model:
            self.model = self.load_model(model_path=self.model_path, tokenizer=self.tokenizer_gen)
            # self.model.train()
            # self.model.eval()
        else:
            self.model = None

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

    def load_tokenizer(
            self,
            model_path,
            padding_side="left",
            truncation_side="left",
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            truncation_side=truncation_side,  # "right" for training, "left" for generating
            cache_dir=self.cache_dir,
        )
        # tokenizer.add_special_tokens({"pad_token": "<|pad_of_text|>"})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        max_len = tokenizer.max_len_single_sentence
        if self.max_seq_len <= 0:
            self.max_seq_len = max_len
        else:
            self.max_seq_len = min(self.max_seq_len, max_len)
        if self.verbose:
            self.logger.info(f">>> len(tokenizer.vocab) = {len(tokenizer.vocab)}; "
                             f"tokenizer.max_len_single_sentence = {max_len}; "
                             f"max_seq_len = {self.max_seq_len}")

        return tokenizer

    def load_model(
            self,
            model_path,
            tokenizer,
    ):
        # Load the model
        # model = AutoModelForCausalLM.from_pretrained(
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # torch.bfloat16
            # torch_dtype=torch.float8_e5m2,  # torch.float8
            device_map="auto",  # !pip install accelerate
            # device_map=self.cuda_dict["device"] if self.debug else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # model = model.to(device=self.cuda_dict["device"])
        # list(model.state_dict().keys())

        model.generation_config.pad_token_id = tokenizer.pad_token_id  # eos_token_id
        # model.generation_config.pad_token_id = self.tokenizer_gen.pad_token_id  # eos_token_id
        # model.language_model.generation_config.pad_token_id = self.tokenizer_gen.pad_token_id

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
            self.logger.info(f">>> Model loaded: {model_path}")
            self.logger.info(f">>> Number of total parameters: {total_params}")  # 10,670,220,835
            self.logger.info(f">>> Number of trainable parameters: {train_params}")

        return model
