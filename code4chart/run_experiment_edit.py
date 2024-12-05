import os
import sys
# import json
import time
from typing import Optional, List, Dict, Tuple, Any

import fire
# import numpy as np

# import base64
# from PIL import Image
# from io import BytesIO

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login as hf_login
# from datasets import load_dataset, DatasetDict, Dataset

from code4chart.text_llm import TextLLM
from code4chart.code_llm import CodeLLM
from code4chart.vlm import VLM
from code4chart.default_inputs import DefaultInputs
from code4chart.init_functions import logger_setup, cuda_setup, random_setup


class Code4ChartExpEdit:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            hf_id_text_llm: str = "meta-llama/Llama-3.1-8B-Instruct",
            hf_id_code_llm: str = "meta-llama/CodeLlama-7b-Instruct-hf",
            hf_id_vlm: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
            bsz: int = 1,
            max_seq_len: int = 1024,
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Does The Plotting Code Improve Chart Understanding for Vision-Language Models?

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param cache_dir: The root directory of the cache.
        :param project_root_dir: The directory of the project root.
        :param hf_id_text_llm: The Hugging Face model ID of Text LLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param hf_id_code_llm: The Hugging Face model ID of Code LLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param hf_id_vlm: The Hugging Face model ID of VLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param bsz: The batch size.
        :param max_seq_len: The maximum sequence length for padding/truncation.
        :param show_generation: Whether to show outputs during generation.
        :param debug: Debugging / developing mode.
        :return: None.
        """

        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.cache_dir = cache_dir
        self.project_root_dir = project_root_dir
        self.home_dir = os.path.expanduser("~")
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.show_generation = show_generation
        self.debug = debug
        self.hf_id_text_llm = hf_id_text_llm
        self.hf_id_code_llm = hf_id_code_llm
        self.hf_id_vlm = hf_id_vlm

        # Data and checkpoint directory
        self.data_dir = os.path.join(project_root_dir, "data/code4chart")
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/code4chart")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.data_dir_raw = os.path.join(self.data_dir, "raw")
        self.data_dir_process = os.path.join(self.data_dir, "process")
        os.makedirs(self.data_dir_raw, exist_ok=True)
        os.makedirs(self.data_dir_process, exist_ok=True)

        # self.text_llm_model = TextLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )
        # self.code_llm_model = CodeLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_code_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )
        # self.vlm_model = VLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_vlm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

    def run_chart_qa_no_code(
            self,
    ) -> None:
        return None

    def run_chart_qa_with_code(
            self,
    ) -> None:
        return None

    # def run_chart_qa_with_code_no_comments(
    #         self,
    # ) -> None:
    #     return None

    def run_chart_cap_no_code(
            self,
    ) -> None:
        return None

    def run_chart_cap_with_code(
            self,
    ) -> None:
        return None

    # def run_chart_cap_with_code_no_comments(
    #         self,
    # ) -> None:
    #     return None


def main(
    task: int = 0,
    verbose: bool = False,
    seed: int = 42,
    cuda: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    hf_id_text_llm: str = "meta-llama/Llama-3.1-8B-Instruct",
    hf_id_code_llm: str = "meta-llama/CodeLlama-7b-Instruct-hf",
    hf_id_vlm: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    bsz: int = 1,
    max_seq_len: int = 1024,
    show_generation: bool = False,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the (robustness) experiment on the edited chart QA and chart captioning tasks with or without code as input.

    :param task: the task of the current run session.
    :param verbose: Verbose mode: show logs.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param hf_id_text_llm: The Hugging Face model ID of Text LLM. Format: ORGANIZATION_NAME/MODEL_NAME
    :param hf_id_code_llm: The Hugging Face model ID of Code LLM. Format: ORGANIZATION_NAME/MODEL_NAME
    :param hf_id_vlm: The Hugging Face model ID of VLM. Format: ORGANIZATION_NAME/MODEL_NAME
    :param bsz: The batch size.
    :param max_seq_len: The maximum sequence length for padding/truncation.
    :param show_generation: Whether to show outputs during generation.
    :param debug: Debugging / developing mode.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Code4Chart")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    c4c_exp = Code4ChartExpEdit(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        hf_id_text_llm=hf_id_text_llm,
        hf_id_code_llm=hf_id_code_llm,
        hf_id_vlm=hf_id_vlm,
        bsz=bsz,
        max_seq_len=max_seq_len,
        show_generation=show_generation,
        debug=debug,
    )

    task = int(task)
    match task:
        case 1:
            c4c_exp.run_chart_qa_no_code()
        case 2:
            c4c_exp.run_chart_qa_with_code()
        case 3:
            c4c_exp.run_chart_cap_no_code()
        case 4:
            c4c_exp.run_chart_cap_with_code()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
