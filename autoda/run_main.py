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

from autoda.text_llm import TextLLM
from autoda.code_llm import CodeLLM
from autoda.vlm import VLM
from autoda.default_inputs import DefaultInputs
from utils.init_functions import logger_setup, cuda_setup, random_setup


class AutoDA:

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
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Automated Data Analysis via Multimodal Large Language Models

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
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
        self.data_dir = os.path.join(project_root_dir, "data/autoda")
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/autoda")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.text_llm_model = TextLLM(
            verbose=verbose, logger=logger, cuda_dict=cuda_dict,
            cache_dir=cache_dir, project_root_dir=project_root_dir,
            hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        )
        self.code_llm_model = CodeLLM(
            verbose=verbose, logger=logger, cuda_dict=cuda_dict,
            cache_dir=cache_dir, project_root_dir=project_root_dir,
            hf_id=hf_id_code_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        )
        self.vlm_model = VLM(
            verbose=verbose, logger=logger, cuda_dict=cuda_dict,
            cache_dir=cache_dir, project_root_dir=project_root_dir,
            hf_id=hf_id_vlm, bsz=bsz, show_generation=show_generation, debug=debug,
        )

    def run(
            self,
            data_csv_path: str,
            user_da_req: Optional[str] = None,
    ):
        # The main process
        assert os.path.isfile(data_csv_path), f"Assertion Error: `data_csv_path` does not exist: {data_csv_path}"
        if isinstance(user_da_req, str) and len(user_da_req) > 0:
            # Case 1: The user has provided data analysis requirements.
            # We use LLMs to analyze the basic info of the data table and the da reqs to provide solutions
            pass
        else:
            # Case 2: Call Text LLMs to analyze the basic info of the data table and
            #   provide potential data analysis requirements for the user to choose (can choose multiple options)
            pass

        return None

    def step1_1_read_csv(
            self,
            data_csv_path: str = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Load the tabular data and run basic analysis (e.g., using Pandas to get some row and column information)
        table_data = None
        table_info = dict({})

        return table_data, table_info

    def step1_2_load_da_req(
            self,
            user_input: str = None,
            table_info: dict = None,
    ) -> List[str]:
        # Load the data analysis requirements/instructions
        # 1. Users specify a requirement -> get this string -> analyze it using LLMs
        # 2. Users choose requirements form the options we provide -> get this string -> analyze it using LLMs
        # 3. Users pass a default/general DA requirement -> use all da_req we have -> analyze it using LLMs
        da_req = [""]

        return da_req

    def step1_3_analyze_da_req(
            self,
            da_req: List[str] = None,
    ) -> List[Dict[str, Any]]:
        # For each da_req, we use Text2Text LLMs to analyze the requirement, provide some solutions or steps,
        #   and give the proper chart types & specifications so that
        #   we can use Code LLMs to generate corresponding visualization code.
        # TODO: there are some recent related work about using LLMs to analyze DA tasks,
        #   e.g., DracoGPT https://arxiv.org/abs/2408.06845
        #   find pre-defined/existing DA tasks/requirements in Draco 2 https://arxiv.org/abs/2308.14247
        da_req_analysis = []  # [dict() for _ in da_req]

        return da_req_analysis

    def step2_1_gen_vis_code(
            self,
            table_data,
            table_info: dict = None,
            da_req: List[str] = None,
            da_req_analysis: List[Dict[str, Any]] = None,
    ) -> List[str]:
        # Generate visualization code (Python3, using matplotlib/seaborn library) per da_req using Code LLMs
        # TODO: Consider expanding key attributes of matplotlib functions (show the default values)
        vis_code_python = []

        return vis_code_python

    def step2_2_execute_vis_code(
            self,
            vis_code_python: List[str] = None,
    ) -> Tuple[List[str], List[Any]]:
        # Execute the visualization code and get the chart figure (the jpg file and its base64 encoding)

        chart_path = []
        chart_base64 = []

        return chart_path, chart_base64

    def step3_gen_chart(
            self,
            table_data,
            table_info: dict = None,
            da_req: List[str] = None,
            da_req_analysis: List[Dict[str, Any]] = None,
            vis_code_python: List[str] = None,
            chart_path: List[str] = None,
            chart_base64: list = None,
    ) -> List[Dict[str, Any]]:
        # Input all information to Chart LLMs and obtain chart captions/descriptions and relevant analysis/insights
        # Here, "chart captions" faithfully/objectively describe the observations in the chart,
        #   while the "relevant analysis" is further insights
        chart_captions = []
        chart_insights = []

        assert len(chart_captions) == len(chart_insights)
        chart_analysis = [{
            "caption": caption,
            "insight": insight,
        } for caption, insight in zip(chart_captions, chart_insights)]

        return chart_analysis

    def step4_overall_analysis(
            self,
            table_data,
            table_info: dict = None,
            da_req: List[str] = None,
            da_req_analysis: List[Dict[str, Any]] = None,
            vis_code_python: List[str] = None,
            chart_path: List[str] = None,
            chart_base64: list = None,
            chart_analysis: List[Dict[str, Any]] = None
    ) -> str:
        # Input all information to Text2Text LLMs and obtain the overall analysis for the tabular data
        overall_analysis = ""

        return overall_analysis


def main(
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
    interactive: bool = False,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the whole data analysis pipeline.

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
    :param interactive: Interactive model (accepting user inputs).
    :param debug: Debugging / developing mode.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("AutoDA_MLLM")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    # hf_login(token="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")

    auto_da = AutoDA(
        verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        cache_dir=cache_dir, project_root_dir=project_root_dir,
        hf_id_text_llm=hf_id_text_llm, hf_id_code_llm=hf_id_code_llm, hf_id_vlm=hf_id_vlm,
        bsz=bsz, show_generation=show_generation, debug=debug,
    )

    def_input = DefaultInputs(project_root_dir=project_root_dir)

    if interactive:
        # Get the filepath of the data csv file
        input_data_csv_path = input("\nPlease input your data csv filepath:\n"
                                    "Press Enter to choose from example datasets.")
        input_data_csv_path = input_data_csv_path.strip()
        if len(input_data_csv_path) == 0:
            input_data_csv_id = input(
                "\n" + "\n".join(f"{k}: {v}" for k, v in def_input.data_csv_dict.items()) +
                "\nPlease input a number: ")
            input_data_csv_id = input_data_csv_id.strip()
            try:
                input_data_csv_id = int(input_data_csv_id)
                input_data_csv_path = def_input.data_csv_dict[input_data_csv_id]
            except Exception as e:
                print(e)
                sys.exit(1)
        assert os.path.isfile(input_data_csv_path), f"Assertion Error: Path does not exist: {input_data_csv_path}"

        # Get the data analysis requirement
        input_user_da_req = input("\nPlease input your data analysis requirements:\n"
                                  "Press Enter to get more specified data analysis suggestions.")
        input_user_da_req = input_user_da_req.strip()
        # if len(input_user_da_req) == 0:
        #     input_user_da_req = input(
        #         "\n" + "\n".join(f"{k}: {v['question']}" for k, v in def_input.da_req_all_dict.items()) +
        #         "\nPlease input a number (Press Enter to get more specified suggestions): ")
        #     input_user_da_req = input_user_da_req.strip()
        #     if len(input_user_da_req) == 0:
        #         input_user_da_req = ""
        #         pass
        #     else:
        #         try:
        #             input_user_da_req_id = int(input_user_da_req)
        #             input_user_da_req = def_input.da_req_all_dict[input_user_da_req_id]
        #         except Exception as e:
        #             print(e)
        #             sys.exit(1)
        input_user_da_req = input_user_da_req.strip()
    else:
        input_data_csv_path = def_input.data_csv_dict[3]  # "data/Iris_Species.csv"
        input_user_da_req = ""  # Let the Text LLM to provide data analysis requirements

    auto_da.run(data_csv_path=input_data_csv_path, user_da_req=input_user_da_req)

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
