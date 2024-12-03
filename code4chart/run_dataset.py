import os
# import sys
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
from utils.init_functions import logger_setup, cuda_setup, random_setup


class Code4ChartDataset:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            data_csv_path_list: List[str],
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            hf_id_text_llm: str = "meta-llama/Llama-3.1-8B-Instruct",
            hf_id_code_llm: str = "meta-llama/CodeLlama-7b-Instruct-hf",
            hf_id_vlm: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
            bsz: int = 1,
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Does The Plotting Code Improve Chart Understanding for Vision-Language Models?

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param data_csv_path_list: The list of data csv paths.
        :param cache_dir: The root directory of the cache.
        :param project_root_dir: The directory of the project root.
        :param hf_id_text_llm: The Hugging Face model ID of Text LLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param hf_id_code_llm: The Hugging Face model ID of Code LLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param hf_id_vlm: The Hugging Face model ID of VLM. Format: ORGANIZATION_NAME/MODEL_NAME
        :param bsz: The batch size.
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
        self.show_generation = show_generation
        self.debug = debug
        self.hf_id_text_llm = hf_id_text_llm
        self.hf_id_code_llm = hf_id_code_llm
        self.hf_id_vlm = hf_id_vlm

        # Data and checkpoint directory
        self.data_csv_path_list = data_csv_path_list
        self.data_dir = os.path.join(project_root_dir, "data/code4chart")
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/code4chart")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

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

    def step1_get_metadata(
            self,
    ) -> str:
        # Load the tabular data and run basic analysis (e.g., using Pandas to get some row and column information)
        metadata = []  # List[Dict[str, Any]]

        # Write the data_csv_path and metadata into jsonl files
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")

        return metadata_fp

    def step2_analyze_da_reqs(
            self,
    ) -> str:
        # For each da_req, we use Text2Text LLMs to analyze the requirement, provide some solutions or steps,
        #   and give the proper chart types & specifications so that
        #   we can use Code LLMs to generate corresponding visualization code.
        # TODO: there are some recent related work about using LLMs to analyze DA tasks,
        #   e.g., DracoGPT https://arxiv.org/abs/2408.06845
        #   find pre-defined/existing DA tasks/requirements in Draco 2 https://arxiv.org/abs/2308.14247
        da_reqs = []  # List[Dict[str, Any]]

        # Load "metadata.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")

        # Load the Text LLM
        # self.text_llm_model = TextLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and da_reqs into jsonl files
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")

        return da_reqs_fp

    def step3_gen_vis_code(
            self,
    ) -> str:
        # Input: data_csv_path: str, metadata: Dict[str, Any], da_reqs: List[Dict[str, Any]]
        # Generate visualization code (Python3, using matplotlib/seaborn library) per da_req using Code LLMs
        # TODO: Consider expanding key attributes of matplotlib functions (show the default values)
        vis_code = []  # List[str], Python3 matplotlib code

        # Get self.data_csv_path_list
        # Load "metadata.jsonl" and "da_reqs.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")

        # Load the Code LLM
        # self.code_llm_model = CodeLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_code_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and vis_code into jsonl files
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")

        return vis_code_fp

    def step4_exec_vis_code(
            self,
    ) -> str:
        # Input: vis_code: List[str]
        # Execute the visualization code and get the chart figure (the jpg file and its base64 encoding)
        chart_path = []  # List[str]
        chart_base64 = []  # List[Base64]

        # Get self.data_csv_path_list
        # Load "vis_code.jsonl"
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")

        # exec(open("file.py").read())
        #
        # filepath = "script2.py"
        # with open(filepath) as fp_in:
        #     exec(fp_in.read())
        #
        # import subprocess
        # subprocess.run(["python3", "script2.py"])

        # Write the data_csv_path, chart_path, and chart_base64 into jsonl files
        chart_image_fp = os.path.join(self.data_dir, "chart_image.jsonl")

        return chart_image_fp

    def step5_chart_cap(
            self,
    ) -> str:
        # Input all information to Chart LLMs and obtain chart captions/descriptions and relevant analysis/insights
        # Here, "chart captions" faithfully/objectively describe the observations in the chart,
        #   while the "relevant analysis" is further insights
        chart_caption = []  # List[str]
        # chart_insight = []  # List[str]

        # Get self.data_csv_path_list
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", and "chart_image.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir, "chart_image.jsonl")

        # Load the Vision-language Model (Multimodal LLM)
        # self.vlm_model = VLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_vlm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # assert len(chart_caption) == len(chart_insight)
        # chart_analysis = [{
        #     "caption": caption,
        #     "insight": insight,
        # } for caption, insight in zip(chart_caption, chart_insight)]  # List[Dict[str, Any]]

        # Write the data_csv_path and chart_caption into jsonl files
        # chart_analysis_fp = os.path.join(self.data_dir, "chart_analysis.jsonl")
        chart_caption_fp = os.path.join(self.data_dir, "chart_caption.jsonl")

        return chart_caption_fp

    def step6_overall_analysis(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and obtain the overall analysis for each table (tabular dataset)
        overall_analysis = []  # List[str]

        # Get self.data_csv_path_list
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", and "chart_caption.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")
        chart_caption_fp = os.path.join(self.data_dir, "chart_caption.jsonl")

        # Load the Text LLM
        # self.text_llm_model = TextLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and overall_analysis into jsonl files
        overall_analysis_fp = os.path.join(self.data_dir, "overall_analysis.jsonl")

        return overall_analysis_fp

    def step7_chart_qa(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and construct a chart QA evaluation benchmark
        # TODO: manually construct a small evaluation dataset of high quality (or as the guide for text LLMs)
        chart_question = []  # List[str]
        chart_option = []  # List[List[str]] -> five options, one with correct answer (consider distractors' quality)
        chart_answer = []  # List[int] -> each entry: the index of correct answer in each option list

        # Get self.data_csv_path_list
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_image.jsonl",
        #   "chart_caption.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir, "chart_image.jsonl")
        chart_caption_fp = os.path.join(self.data_dir, "chart_caption.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir, "overall_analysis.jsonl")

        # Write each chart QA example into jsonl files
        res_fp = os.path.join(self.data_dir, "c4c_qa.jsonl")

        # To upload to Hugging Face datasets

        return res_fp

    def step8_chart_cap(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and construct a chart captioning evaluation benchmark
        # TODO: manually construct a small evaluation dataset of high quality (or as the guide for text LLMs)
        chart_question = []  # List[str]
        chart_option = []  # List[List[str]] -> five options, one with correct answer (consider distractors' quality)
        chart_answer = []  # List[int] -> each entry: the index of correct answer in each option list

        # Get self.data_csv_path_list
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_image.jsonl",
        #   "chart_caption.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir, "chart_image.jsonl")
        chart_caption_fp = os.path.join(self.data_dir, "chart_caption.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir, "overall_analysis.jsonl")

        # Write each chart captioning example into jsonl files
        res_fp = os.path.join(self.data_dir, "c4c_cap.jsonl")

        # To upload to Hugging Face datasets

        return res_fp


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
    show_generation: bool = False,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the whole data analysis pipeline.

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

    # hf_login(token="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")

    def_input = DefaultInputs(project_root_dir=project_root_dir)

    c4c_data = Code4ChartDataset(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        data_csv_path_list=def_input.data_csv_list,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        hf_id_text_llm=hf_id_text_llm,
        hf_id_code_llm=hf_id_code_llm,
        hf_id_vlm=hf_id_vlm,
        bsz=bsz,
        show_generation=show_generation,
        debug=debug,
    )

    task = int(task)
    match task:
        case 1:
            c4c_data.step1_get_metadata()
        case 2:
            c4c_data.step2_analyze_da_reqs()
        case 3:
            c4c_data.step3_gen_vis_code()
        case 4:
            c4c_data.step4_exec_vis_code()
        case 5:
            c4c_data.step5_chart_cap()
        case 6:
            c4c_data.step6_overall_analysis()
        case 7:
            c4c_data.step7_chart_qa()
        case 8:
            c4c_data.step8_chart_cap()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
