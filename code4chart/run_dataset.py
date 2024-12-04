import os
# import sys
import json
import time
from typing import Optional, List, Dict, Tuple, Any

import fire
import numpy as np
import pandas as pd

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
from utils.numpy_encoder import NumpyEncoder


class Code4ChartDataset:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            datasets_info: List[Dict[str, Any]],
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            hf_id_text_llm: str = "meta-llama/Llama-3.1-8B-Instruct",
            hf_id_code_llm: str = "meta-llama/CodeLlama-7b-Instruct-hf",
            hf_id_vlm: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
            bsz: int = 1,
            max_seq_len: int = 512,
            show_generation: bool = False,
            debug: bool = False,
    ):
        """
        Does The Plotting Code Improve Chart Understanding for Vision-Language Models?

        :param verbose: Verbose mode: show logs.
        :param logger: The logger to show logs.
        :param cuda_dict: The cuda/GPU information dictionary.
        :param datasets_info: The list of data csv paths.
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
        self.show_generation = show_generation
        self.debug = debug
        self.hf_id_text_llm = hf_id_text_llm
        self.hf_id_code_llm = hf_id_code_llm
        self.hf_id_vlm = hf_id_vlm

        # Data and checkpoint directory
        self.datasets_info = datasets_info
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

    def step1_get_metadata(
            self,
    ) -> str:
        # Load the tabular data and run basic analysis (e.g., using Pandas to get some row and column information)
        metadata = []  # List[Dict[str, Any]]

        # Get the basic statistics on the dataset
        for ds in self.datasets_info:
            ds_metadata = dict()

            # 1. Dataset-wise:
            #   # of examples (rows), # of features (cols)
            ds_id, ds_url, ds_desc = ds["id"], ds["url"], ds["description"]
            ds_fp, ds_fn, ds_name = ds["filepath"], ds["filename"], ds["name"]
            assert os.path.isfile(ds_fp)

            ds_metadata["id"] = ds_id
            ds_metadata["url"] = ds_url
            ds_metadata["description"] = ds_desc
            ds_metadata["filepath"] = ds_fp
            ds_metadata["filename"] = ds_fn
            ds_metadata["name"] = ds_name

            df = pd.read_csv(ds_fp)
            num_row, num_col = df.shape
            ds_metadata["num_row"] = num_row
            ds_metadata["num_col"] = num_col

            feat_list = list(df.columns)
            ds_metadata["features"] = []
            for feat in feat_list:
                # 2. Feature-wise:
                #   the feature name, datatype, # of miss/valid/unique values, the max/min/mean/std of numerical values
                cur_feat_dict = dict()
                cur_feat_dict["name"] = feat

                df_feat = df[feat]
                num_total = len(df_feat)
                num_miss = int(df_feat.isna().sum())
                # num_valid = num_row - num_miss
                df_feat = df_feat.dropna(axis=0)
                num_valid = len(df_feat)
                assert num_total == num_miss + num_valid
                num_unique = len(df_feat.unique())

                cur_feat_dict["num_total"] = num_total
                cur_feat_dict["num_miss"] = num_miss
                cur_feat_dict["num_valid"] = num_valid
                cur_feat_dict["num_unique"] = num_unique

                # Convert the dtype of the current feature col, and do statistics on numerical data
                numerical_stat = dict()
                cur_dtype = "object"
                astype_flag = False
                try:
                    df_feat = df_feat.astype("float32")
                    astype_flag = True
                    cur_dtype = "float32"
                    numerical_stat = {
                        "num": num_valid,
                        "min": np.min(df_feat).item(),
                        "max": np.max(df_feat).item(),
                        "mean": np.mean(df_feat).item(),
                        "std": np.std(df_feat).item(),
                    }
                except ValueError:
                    pass

                if not astype_flag:
                    try:
                        df_feat = df_feat.astype("int64")
                        astype_flag = True
                        cur_dtype = "int64"
                        numerical_stat = {
                            "num": num_valid,
                            "min": np.min(df_feat).item(),
                            "max": np.max(df_feat).item(),
                            "mean": np.mean(df_feat).item(),
                            "std": np.std(df_feat).item(),
                        }
                    except ValueError:
                        pass

                if not astype_flag:
                    try:
                        df_feat = df_feat.astype("string")
                        # df_feat = df_feat.astype(str)
                        astype_flag = True
                        cur_dtype = "str"
                    except ValueError:
                        pass

                # if not astype_flag:
                #     pass
                # cur_dtype = df_feat.dtype

                cur_feat_dict["dtype"] = cur_dtype
                cur_feat_dict["numerical_stat"] = numerical_stat

                # Save the metadata of the current feature
                ds_metadata["features"].append(cur_feat_dict)

            # Save the metadata of the current dataset
            metadata.append(ds_metadata)

        # Write the data_csv_path and metadata into jsonl files
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        write_cnt = 0
        with open(metadata_fp, "w", encoding="utf-8") as fp_out:
            for _item in metadata:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {metadata_fp}")
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

        # Load the metadata
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]

        # Load the Text LLM
        text_llm = TextLLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_text_llm, bsz=self.bsz, show_generation=self.show_generation, debug=self.debug,
        )

        da_reqs = []  # List[Dict[str, Any]]
        for metadata_dict in metadata:
            # Based on the metadata of the datasets, ask the Text LLM to generate some reasonable
            #   data analysis requirements (da_reqs) and corresponding visualization chart (chart type).
            cur_reqs_dict = dict()
            cur_reqs_dict["id"] = metadata_dict["id"]

            prompt_list = []
            req_list = []
            prompt_overall = f"""
Please construct a data analysis requirement based on the dataset information as follows. \
Each data analysis requirement should include a visualization instruction, a specific chart type for visualization, \
and related data table columns (features).

## Dataset Name: {metadata_dict["name"]}

## Dataset Description:
{metadata_dict["description"]}

## Dataset Statistics:
- Number of rows (dataset size): {metadata_dict["num_row"]}
- Number of columns (features): {metadata_dict["num_col"]}

## Dataset Features (Columns):
- Feature Names: {", ".join([feat_dict["name"] for feat_dict in metadata_dict["features"]])}
- Feature Data Types: {", ".join([feat_dict["dtype"] for feat_dict in metadata_dict["features"]])}

## Data analysis requirement:
            """.strip()

            prompt_list.append(prompt_overall)

            for feat_dict in metadata_dict["features"]:
                # Here, we only use the non-NAN information of one single feature
                #   TODO: future work: analyze multiple features (such as the correlation between two features)
                # num_total, num_miss = feat_dict["num_total"], feat_dict["num_miss"]
                num_valid, num_unique = feat_dict["num_valid"], feat_dict["num_unique"]
                cur_dtype, numerical_stat = feat_dict["dtype"], feat_dict["numerical_stat"]

                prompt_feature = f"""
Please construct a data analysis requirement based on the feature information as follows. \
Each data analysis requirement should include a visualization instruction, a specific chart type for visualization, \
and related data table columns (features).

## Dataset Name: {metadata_dict["name"]}

## Feature Information:
- Feature Name: {feat_dict["name"]}
- Data Type: {cur_dtype}
- Number of all rows (feature values): {num_valid}
- Number of unique feature values: {num_unique}
                """.strip()
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    prompt_feature += "\n\n" + f"""
## Numerical Values Statistics:
- Min: {numerical_stat["min"]:.2f}
- Max: {numerical_stat["max"]:.2f}
- Mean: {numerical_stat["mean"]:.2f}
- Std: {numerical_stat["std"]:.2f}

## Data analysis requirement:
                    """.strip()
                else:
                    prompt_feature += "\n\n## Data analysis requirement:"

                prompt_list.append(prompt_feature)

            for prompt in prompt_list:
                gen_dict = text_llm.run_generation(
                    prompts=[prompt], model=text_llm.model, tokenizer=text_llm.tokenizer_gen,
                    need_tokenize=True, max_new_tokens=100,
                    temperature=0.1, top_p=0.1,  # Be more deterministic when choosing an option
                )
                output_text = gen_dict["output_text"][0].strip()
                req_list.append(output_text)  # TODO: save the prompt too

            cur_reqs_dict["prompts"] = prompt_list
            cur_reqs_dict["da_reqs"] = req_list
            da_reqs.append(cur_reqs_dict)

        # Write the data_csv_path and da_reqs into jsonl files
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        write_cnt = 0
        with open(da_reqs_fp, "w", encoding="utf-8") as fp_out:
            for _item in da_reqs_fp:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {da_reqs_fp}")
        return da_reqs_fp

    def step3_gen_vis_code(
            self,
    ) -> str:
        # Input: data_csv_path: str, metadata: Dict[str, Any], da_reqs: List[Dict[str, Any]]
        # Generate visualization code (Python3, using matplotlib/seaborn library) per da_req using Code LLMs
        # TODO: Consider expanding key attributes of matplotlib functions (show the default values)
        vis_code = []  # List[str], Python3 matplotlib code

        # Get self.datasets_info
        # Load "metadata.jsonl" and "da_reqs.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")

        # Load the Code LLM
        # self.code_llm_model = CodeLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_code_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and vis_code into jsonl files
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")

        return vis_code_fp

    def step4_exec_vis_code(
            self,
    ) -> str:
        # Input: vis_code: List[str]
        # Execute the visualization code and get the chart figure (the jpg file and its base64 encoding)
        chart_path = []  # List[str]
        chart_base64 = []  # List[Base64]

        # Get self.datasets_info
        # Load "vis_code.jsonl"
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")

        # exec(open("file.py").read())
        #
        # filepath = "script2.py"
        # with open(filepath) as fp_in:
        #     exec(fp_in.read())
        #
        # import subprocess
        # subprocess.run(["python3", "script2.py"])

        # Write the data_csv_path, chart_path, and chart_base64 into jsonl files
        chart_image_fp = os.path.join(self.data_dir_process, "chart_image.jsonl")

        return chart_image_fp

    def step5_chart_cap(
            self,
    ) -> str:
        # Input all information to Chart LLMs and obtain chart captions/descriptions and relevant analysis/insights
        # Here, "chart captions" faithfully/objectively describe the observations in the chart,
        #   while the "relevant analysis" is further insights
        chart_caption = []  # List[str]
        # chart_insight = []  # List[str]

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", and "chart_image.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir_process, "chart_image.jsonl")

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
        # chart_analysis_fp = os.path.join(self.data_dir_process, "chart_analysis.jsonl")
        chart_caption_fp = os.path.join(self.data_dir_process, "chart_caption.jsonl")

        return chart_caption_fp

    def step6_overall_analysis(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and obtain the overall analysis for each table (tabular dataset)
        overall_analysis = []  # List[str]

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", and "chart_caption.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_caption_fp = os.path.join(self.data_dir_process, "chart_caption.jsonl")

        # Load the Text LLM
        # self.text_llm_model = TextLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and overall_analysis into jsonl files
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        return overall_analysis_fp

    def step7_chart_qa(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and construct a chart QA evaluation benchmark
        # TODO: manually construct a small evaluation dataset of high quality (or as the guide for text LLMs)
        chart_question = []  # List[str]
        chart_option = []  # List[List[str]] -> five options, one with correct answer (consider distractors' quality)
        chart_answer = []  # List[int] -> each entry: the index of correct answer in each option list

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_image.jsonl",
        #   "chart_caption.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir_process, "chart_image.jsonl")
        chart_caption_fp = os.path.join(self.data_dir_process, "chart_caption.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        # Write each chart QA example into jsonl files
        res_fp = os.path.join(self.data_dir_process, "c4c_qa.jsonl")

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

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_image.jsonl",
        #   "chart_caption.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_image_fp = os.path.join(self.data_dir_process, "chart_image.jsonl")
        chart_caption_fp = os.path.join(self.data_dir_process, "chart_caption.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        # Write each chart captioning example into jsonl files
        res_fp = os.path.join(self.data_dir_process, "c4c_cap.jsonl")

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
    max_seq_len: int = 512,
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

    # hf_login(token="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")

    def_input = DefaultInputs(project_root_dir=project_root_dir)

    c4c_data = Code4ChartDataset(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        datasets_info=def_input.datasets_info,
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