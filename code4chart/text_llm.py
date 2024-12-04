import os
# import sys
# import json
# import time
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextLLM:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            bsz: int = 1,
            max_seq_len: int = 512,
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Text-to-Text LLM

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param cache_dir: The root directory of the cache.
        :param project_root_dir: The directory of the project root.
        :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
        :param bsz: The batch size.
        :param max_seq_len: The maximum sequence length for padding/truncation.
        :param show_generation: Whether to show outputs during generation.
        :param debug: Debugging / developing mode.
        :return: None.
        """

        # https://arxiv.org/abs/2407.21783
        # "meta-llama/Llama-3.2-1B-Instruct"  "meta-llama/Llama-3.2-3B-Instruct"  "meta-llama/Llama-3.1-8B-Instruct"

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
        self.model = self.load_model(model_path=self.model_path, tokenizer=self.tokenizer_gen)
        # self.model.train()
        # self.model.eval()

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
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # torch.bfloat16
            # torch_dtype=torch.float8_e5m2,  # torch.float8
            device_map="auto",  # !pip install accelerate
            # device_map=self.cuda_dict["device"] if self.debug else "auto",
            # device_map=self.device_mps if self.debug else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # model = model.to(device=self.cuda_dict["device"])
        # list(model.state_dict().keys())
        model.generation_config.pad_token_id = tokenizer.pad_token_id  # eos_token_id
        # model.resize_token_embeddings(len(self.tokenizer_train))  # if added new special tokens (Option 1)
        # model.train()
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f">>> Model loaded: {model_path}")
            self.logger.info(f">>> Number of total parameters: {total_params}")
            self.logger.info(f">>> Number of trainable parameters: {train_params}")

        return model

    def run_generation(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
            max_new_tokens: int = 50,
            temperature: float = 0.0,
            top_p: float = 1.0,
    ) -> dict:
        if need_tokenize:
            input_ids = tokenizer(
                prompts,
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
                return_tensors="pt",
            ).to(model.device)  # Batch tokenization
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        # len_input = input_ids.data["input_ids"].size(-1)

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            assert max_new_tokens > 0
            outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.terminators_gen,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                # output_attentions=False,
                # output_hidden_states=False,
                # output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
        output_ids = outputs["sequences"]
        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(prompts) == len(output_text)
        output_text_pure = []
        for _input, _prompt, _output in zip(input_text, prompts, output_text):
            output_pure = _output[len(_input):]
            output_text_pure.append(output_pure)
            if self.verbose and self.show_generation:
                # self.logger.info(f"\n\n\n================================== >>> [Batch: {cur_batch}] <<<\n")
                # self.logger.info("================================== >>> input (raw) <<<")
                # self.logger.info(_input)
                # self.logger.info("================================== >>> prompt <<<")
                # self.logger.info(_prompt)
                self.logger.info("================================== >>> output <<<")
                self.logger.info(output_pure)

        return {
            "prompts": prompts,
            "input_ids": input_ids,
            "input_text": input_text,
            "outputs": outputs,
            "output_ids": output_ids,
            # "output_text": output_text,
            "output_text": output_text_pure,
        }

    def run_language_modeling(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
    ) -> dict:
        if need_tokenize:
            input_ids = tokenizer(
                prompts,
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
                return_tensors="pt",
            ).to(model.device)  # Batch tokenization
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        # len_input = input_ids.data["input_ids"].size(-1)
        target_ids = input_ids["input_ids"].to(model.device)
        input_ids.data["labels"] = target_ids

        with torch.no_grad():
            outputs = model(
                **input_ids,
                # labels=target_ids,
                # output_attentions=False,
                # output_hidden_states=False,
                # output_scores=True,
                # output_logits=True,
                # return_dict_in_generate=True,
            )
        output_ids = outputs["logits"].argmax(-1)
        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(prompts) == len(output_text)
        output_text_pure = []
        for _input, _prompt, _output in zip(input_text, prompts, output_text):
            output_pure = _output[len(_input):]
            output_text_pure.append(output_pure)
            if self.verbose and self.show_generation:
                # self.logger.info(f"\n\n\n================================== >>> [Batch: {cur_batch}] <<<\n")
                # self.logger.info("================================== >>> input (raw) <<<")
                # self.logger.info(_input)
                # self.logger.info("================================== >>> prompt <<<")
                # self.logger.info(_prompt)
                self.logger.info("================================== >>> output <<<")
                self.logger.info(output_pure)

        return {
            "prompts": prompts,
            "input_ids": input_ids,
            "input_text": input_text,
            "outputs": outputs,
            "output_ids": output_ids,
            # "output_text": output_text,
            "output_text": output_text_pure,
        }
