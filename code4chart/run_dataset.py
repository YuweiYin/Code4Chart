import os
import sys
import json
import time
from typing import Optional, List, Dict, Tuple, Any

import fire
import numpy as np
import pandas as pd

import base64
from PIL import Image
from io import BytesIO

import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login as hf_login
# from datasets import load_dataset, DatasetDict, Dataset

from code4chart.text_llm import TextLLM
from code4chart.code_llm import CodeLLM
from code4chart.vlm import VLM
from code4chart.default_inputs import DefaultInputs

from code4chart.init_functions import logger_setup, cuda_setup, random_setup
from code4chart.numpy_encoder import NumpyEncoder


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
        self.max_seq_len = max_seq_len
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

        # Write the metadata into jsonl files
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
        # For each dataset, we use Text LLMs to analyze the DA requirement, including visualization instruction
        #   and chart type specifications for the Code LLMs to generate visualization code later.

        # Load the metadata
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]

        # Load the Text LLM
        text_llm = TextLLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_text_llm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
        )

        da_reqs = []  # List[Dict[str, Any]]
        for metadata_dict in metadata:
            # Based on the metadata of the datasets, ask the Text LLM to generate some reasonable
            #   data analysis requirements (da_reqs) and corresponding visualization chart (chart type).
            cur_reqs_dict = dict()
            cur_reqs_dict["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            prompt_list = []
            req_list = []
            for feat_dict in metadata_dict["features"]:
                # Here, we only use the non-NAN information of one single feature
                #   TODO: future work: analyze multiple features (such as the correlation between two features)
                # if self.verbose:
                #     self.logger.info(f">>> >>> Feature: {feat_dict['name']}")
                # num_total, num_miss = feat_dict["num_total"], feat_dict["num_miss"]
                num_valid, num_unique = feat_dict["num_valid"], feat_dict["num_unique"]
                cur_dtype, numerical_stat = feat_dict["dtype"], feat_dict["numerical_stat"]

                prompt_feature = f"""
Dataset Information:
- Dataset Name: {metadata_dict["name"]}
- All Features: {", ".join([x["name"] for x in metadata_dict["features"]])}

Current Feature Information:
- Feature Name: {feat_dict["name"]}
- Data Type: {cur_dtype}
- Number of all rows (feature values): {num_valid}
- Number of unique feature values: {num_unique}
                """.strip()
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    prompt_feature += "\n" + f"""
- Min of Feature Values: {numerical_stat["min"]:.2f}
- Max of Feature Values: {numerical_stat["max"]:.2f}
- Mean of Feature Values: {numerical_stat["mean"]:.2f}
- Std of Feature Values: {numerical_stat["std"]:.2f}
                    """.strip()

                prompt_feature += "\n\n" + f"""
## Task: Please construct one data analysis requirement based on the dataset and feature information above. \
The requirement should include a visualization instruction and specify a chart type for visualization. \
The requirement is to ask models to generate Python3 code using the matplotlib, numpy, and pandas packages \
to plot a chart and save the figure. Be concise, clear, and short.
                """.strip()

                prompt_list.append(prompt_feature)

            for prompt in prompt_list:
                gen_dict = text_llm.run_generation(
                    prompts=[prompt], model=text_llm.model, tokenizer=text_llm.tokenizer_gen,
                    need_tokenize=True, max_new_tokens=512,
                    temperature=0.1, top_p=0.1,  # Be more deterministic when choosing an option
                )
                output_text = gen_dict["output_text"][0].strip()
                req_list.append(output_text)

            cur_reqs_dict["prompts"] = prompt_list
            cur_reqs_dict["da_reqs"] = req_list
            da_reqs.append(cur_reqs_dict)
            if self.debug:
                sys.exit(0)

        # Write the data analysis requirements into jsonl files
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        write_cnt = 0
        with open(da_reqs_fp, "w", encoding="utf-8") as fp_out:
            for _item in da_reqs:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {da_reqs_fp}")
        # Total Running Time: 2653.8 sec (44.2 min)
        return da_reqs_fp

    def step3_gen_vis_code(
            self,
    ) -> str:
        # For each da_req, we use Code LLMs to generate Python3 code to plot charts.

        # Load the metadata and da_reqs
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(da_reqs_fp, "r", encoding="utf-8") as fp_in:
            da_reqs = [json.loads(line.strip()) for line in fp_in]
        assert isinstance(metadata, list) and isinstance(da_reqs, list) and len(metadata) == len(da_reqs)

        # Load the Code LLM
        code_llm = CodeLLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_code_llm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
        )

        vis_code = []  # List[Dict[str, Any]], Python3 matplotlib code
        for metadata_dict, cur_reqs_dict in zip(metadata, da_reqs):
            # Based on the metadata and da_reqs, ask the Code LLM to generate visualization code (Python3 matplotlib).
            cur_vis_code_dict = dict()
            cur_vis_code_dict["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            # cur_csv_path = metadata_dict["filepath"]
            # assert os.path.isfile(cur_csv_path)
            # df = pd.read_csv(cur_csv_path)

            req_list = cur_reqs_dict["da_reqs"]
            # req_prompt_list = cur_reqs_dict["prompts"]
            code_prompt_list = []
            # vis_data_list = []
            vis_feat_list = []
            assert len(req_list) == len(metadata_dict["features"])
            for req, feat_dict in zip(req_list, metadata_dict["features"]):
                # Here, we only deal with each column (feature) as the whole table can be too large.
                #   TODO: future work: deal with the whole table
                # if self.verbose:
                #     self.logger.info(f">>> >>> Feature: {feat_dict['name']}")
                num_valid, num_unique = feat_dict["num_valid"], feat_dict["num_unique"]
                cur_dtype, numerical_stat = feat_dict["dtype"], feat_dict["numerical_stat"]

                # df_feat = df[feat_dict["name"]]
                # df_feat = df_feat.dropna(axis=0)
                # data_feat = df_feat.tolist()

                cur_code_prompt = f"""
Dataset Information:
- Dataset Name: {metadata_dict["name"]}
- All Features: {", ".join([x["name"] for x in metadata_dict["features"]])}

Current Feature Information:
- Feature Name: {feat_dict["name"]}
- Data Type: {cur_dtype}
- Number of all rows (feature values): {num_valid}
- Number of unique feature values: {num_unique}
                """.strip()
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    cur_code_prompt += "\n" + f"""
- Min of Feature Values: {numerical_stat["min"]:.2f}
- Max of Feature Values: {numerical_stat["max"]:.2f}
- Mean of Feature Values: {numerical_stat["mean"]:.2f}
- Std of Feature Values: {numerical_stat["std"]:.2f}
                    """.strip()

                cur_code_prompt += "\n\n" + f"""
Data Analysis Requirement:
{req}

## Task: Based on the above dataset information and data analysis requirement, \
generate executable Python3 code using the matplotlib, numpy, and pandas packages \
to plot a chart and save the figure. Please only generate executable Python3 code and then stop generation. \
Be concise, clear, and short. \
Assume you can access the data table and target column (list) by the following Python3 code:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("{metadata_dict["filename"]}")
column = data["{feat_dict["name"]}"].tolist()
```

Python3 Code for Chart Plotting:
                """.strip()
                # ## Data Column Values:
                # {data_feat}

                code_prompt_list.append(cur_code_prompt)
                # vis_data_list.append(data_feat)
                vis_feat_list.append(feat_dict["name"])

            vis_code_list = []
            for prompt in code_prompt_list:
                gen_dict = code_llm.run_generation(
                    prompts=[prompt], model=code_llm.model, tokenizer=code_llm.tokenizer_gen,
                    need_tokenize=True, max_new_tokens=512,
                    temperature=0.1, top_p=0.1,  # Be more deterministic when choosing an option
                )
                output_text = gen_dict["output_text"][0].strip()
                vis_code_list.append(output_text)

            cur_vis_code_dict["vis_feat"] = vis_feat_list
            cur_vis_code_dict["prompts"] = code_prompt_list
            cur_vis_code_dict["vis_code"] = vis_code_list
            vis_code.append(cur_vis_code_dict)
            if self.debug:
                sys.exit(0)

        # Write the visualization code into jsonl files
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        write_cnt = 0
        with open(vis_code_fp, "w", encoding="utf-8") as fp_out:
            for _item in vis_code:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {vis_code_fp}")
        # Total Running Time: 2190.3 sec (36.5 min)
        return vis_code_fp

    def step4_vis_code_postprocess(
            self,
    ) -> str:
        # Save the generated visualization code (Python3) into py files and then
        #   perform manual post-processing to make sure only Python code (and comments) remains:
        #   1. Extract the code snippet from the generated content;
        #   2. Ensure the csv data loading path is correct;
        #   3. Change both `plt.show()` and `plt.savefig(...)` to `plt.show()`,
        #     and in the next step, it will be replaced to `plt.savefig(x)`, where x is a proper chart filepath
        # We do not perform extra post-processing to make sure the code is executable.
        #   If it is not, we will skip exec this code for now.

        # Load the metadata and visualization code
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(vis_code_fp, "r", encoding="utf-8") as fp_in:
            vis_code = [json.loads(line.strip()) for line in fp_in]
        assert isinstance(metadata, list) and isinstance(vis_code, list) and len(metadata) == len(vis_code)

        vis_code_post = []  # List[Dict[str, Any]]
        for metadata_dict, cur_vis_code_dict in zip(metadata, vis_code):
            # Based on the metadata and da_reqs, ask the Code LLM to generate visualization code (Python3 matplotlib).
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            vis_feat_list = cur_vis_code_dict["vis_feat"]
            code_prompt_list = cur_vis_code_dict["prompts"]
            vis_code_list = cur_vis_code_dict["vis_code"]

            assert len(vis_feat_list) == len(vis_code_list) == len(code_prompt_list) == len(metadata_dict["features"])
            cur_code_save_dir = os.path.join(self.data_dir_process, "chart_code", str(metadata_dict["id"]))
            os.makedirs(cur_code_save_dir, exist_ok=True)
            code_id = 0
            cur_vis_code_dict["code_filepath"] = []
            cur_vis_code_dict["code_id"] = []
            for feat_name, cur_vis_code, cur_code_prompt, feat_dict in zip(
                    vis_feat_list, vis_code_list, code_prompt_list, metadata_dict["features"]):
                code_id += 1
                cur_code_save_fp = os.path.join(
                    cur_code_save_dir, f"{code_id}-{feat_name.replace('/', '_').strip()}.py")
                # cur_code_str = cur_vis_code
                cur_code_str = cur_code_prompt + cur_vis_code  # We consider both the prompt and the generated vis code
                with open(cur_code_save_fp, "w", encoding="utf-8") as fp_out:
                    fp_out.write(cur_code_str + "\n")
                cur_vis_code_dict["code_filepath"].append(cur_code_save_fp)
                cur_vis_code_dict["code_id"].append(code_id)

            vis_code_post.append(cur_vis_code_dict)

        # Write the chart figures info into jsonl files
        vis_code_post_fp = os.path.join(self.data_dir_process, "vis_code_post.jsonl")
        write_cnt = 0
        with open(vis_code_post_fp, "w", encoding="utf-8") as fp_out:
            for _item in vis_code_post:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {vis_code_post_fp}")
        return vis_code_post_fp

    def step5_exec_vis_code(
            self,
    ) -> str:
        # Execute the generated Python3 visualization code (after post-processing) and plot the chart figures

        # Load the metadata and visualization code
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        vis_code_post_fp = os.path.join(self.data_dir_process, "vis_code_post.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(vis_code_post_fp, "r", encoding="utf-8") as fp_in:
            vis_code_post = [json.loads(line.strip()) for line in fp_in]
        assert isinstance(metadata, list) and isinstance(vis_code_post, list) and len(metadata) == len(vis_code_post)

        chart_figures = []  # List[Dict[str, Any]]
        # chart_base64 = []  # List[Base64]
        done_cnt_all, miss_cnt_all = 0, 0
        num_python_all, num_empty_all, num_comments_all = [], [], []
        for metadata_dict, cur_vis_code_dict in zip(metadata, vis_code_post):
            # Based on the metadata and da_reqs, ask the Code LLM to generate visualization code (Python3 matplotlib).
            cur_chart = dict()
            cur_chart["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            cur_csv_path = metadata_dict["filepath"]
            assert os.path.isfile(cur_csv_path)

            vis_feat_list = cur_vis_code_dict["vis_feat"]
            # code_prompt_list = cur_vis_code_dict["prompts"]
            # vis_code_list = cur_vis_code_dict["vis_code"]
            code_filepath = cur_vis_code_dict["code_filepath"]

            assert len(vis_feat_list) == len(code_filepath) == len(metadata_dict["features"])
            cur_fig_save_dir = os.path.join(self.data_dir_process, "chart_figure", str(metadata_dict["id"]))
            os.makedirs(cur_fig_save_dir, exist_ok=True)
            fig_id = 0
            cur_chart["fig_id"] = []
            cur_chart["fig_filepath"] = []
            cur_chart["fig_base64"] = []
            cur_chart["vis_code"] = []
            cur_chart["code_stat"] = []
            done_cnt, miss_cnt = 0, 0
            for feat_name, vis_code_fp, feat_dict in zip(vis_feat_list, code_filepath, metadata_dict["features"]):
                fig_id += 1
                cur_fig_save_fp = os.path.join(
                    cur_fig_save_dir, f"{fig_id}-{feat_name.replace('/', '_').strip()}.png")

                assert os.path.isfile(vis_code_fp), f"Assertion Error: File does not exist {vis_code_fp}"
                with open(vis_code_fp, "r", encoding="utf-8") as fp_in:
                    # exec(fp_in.read())
                    code_lines = fp_in.readlines()
                new_code_lines = []
                has_savefig = False
                for cur_line in code_lines:
                    if "pd.read_csv" in cur_line:  # Make sure the csv filepath is correct
                        # new_line = f"data = pd.read_csv('{cur_csv_path}')\ndf = data\n"
                        new_line = f"data = pd.read_csv('{cur_csv_path}')\n"
                    elif "plt.savefig" in cur_line:  # Make sure the savefig filepath is correct
                        new_line = f"plt.savefig('{cur_fig_save_fp}', dpi=300, bbox_inches='tight')\n"
                        has_savefig = True
                    elif "plt.show" in cur_line:  # Do not show the figure
                        new_line = "\n"
                    elif "```" in cur_line:  # Clear the markdown lines
                        new_line = "\n"
                    else:
                        new_line = cur_line
                    new_code_lines.append(new_line)
                if not has_savefig:  # Make sure we save the figure
                    new_code_lines.append(f"plt.savefig('{cur_fig_save_fp}', dpi=300, bbox_inches='tight')\n")
                new_code = "".join(new_code_lines)
                cur_chart["vis_code"].append(new_code)
                cur_chart["fig_id"].append(fig_id)
                cur_chart["fig_filepath"].append(cur_fig_save_fp)

                # Code statistics (the number of Python code lines, empty lines, and comment lines)
                num_python, num_empty, num_comments = 0, 0, 0
                for new_line in new_code_lines:
                    new_line = new_line.strip()
                    if len(new_line) == 0:
                        num_empty += 1
                    elif new_line.startswith("#"):
                        num_comments += 1
                    else:
                        num_python += 1
                cur_chart["code_stat"].append({
                    "num_python": num_python,
                    "num_empty": num_empty,
                    "num_comments": num_comments,
                })
                num_python_all.append(num_python)
                num_empty_all.append(num_empty)
                num_comments_all.append(num_comments)

                try:
                    exec(new_code)
                    assert os.path.isfile(cur_fig_save_fp)

                    # Base64 encoding
                    with open(cur_fig_save_fp, "rb") as img_fp_in:
                        img_base64 = base64.b64encode(img_fp_in.read())
                    img_base64_str = img_base64.decode("utf-8")
                    cur_chart["fig_base64"].append(img_base64_str)

                    # # Base64 string to bytes to image
                    # img_base64_bytes = img_base64_str.encode("utf-8")
                    # im = Image.open(BytesIO(base64.b64decode(img_base64_bytes)))
                    # im.save("test_image.png", "PNG")

                    done_cnt += 1
                except Exception as e:
                    if self.verbose:
                        self.logger.info(f">>> >>> Exception: {e} --- Miss file: {cur_fig_save_fp}")
                    cur_chart["fig_base64"].append("")
                    miss_cnt += 1
                    continue

            done_cnt_all += done_cnt
            miss_cnt_all += miss_cnt
            if self.verbose:
                self.logger.info(f">>> >>> Done [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}. "
                                 f"done_cnt={done_cnt}, miss_cnt={miss_cnt}")
            chart_figures.append(cur_chart)

        # Done all, show statistics
        if self.verbose:
            self.logger.info(f">>> >>> Done All. Statistics: "
                             f"done_cnt_all={done_cnt_all}, miss_cnt_all={miss_cnt_all}; "
                             f"avg_num_python={np.mean(num_python_all)}, "
                             f"avg_num_empty={np.mean(num_empty_all)}, "
                             f"avg_num_comments={np.mean(num_comments_all)}")
            # Done All. Statistics: done_cnt_all=63, miss_cnt_all=53;
            # avg_num_python=11.2845, avg_num_empty=4.5948, avg_num_comments=2.6207

        # Write the chart figures info into jsonl files
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")
        write_cnt = 0
        with open(chart_figures_fp, "w", encoding="utf-8") as fp_out:
            for _item in chart_figures:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {chart_figures_fp}")
        # Total Running Time: 378.2 sec (6.3 min)
        return chart_figures_fp

    def step6_chart_cap_gen(
            self,
    ) -> str:
        # For each chart, we use Vision-language models (VLMs) to generate the chart captions (descriptions).
        # TODO: future work: the raw data is not provided in the prompt or generated code
        #   since there are too many values in a column,
        #   but maybe it is helpful for the VLMs to perform chart understanding, especially for knowing the numbers

        # Load the metadata, vis_code, and chart figures
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(vis_code_fp, "r", encoding="utf-8") as fp_in:
            vis_code = [json.loads(line.strip()) for line in fp_in]
        with open(chart_figures_fp, "r", encoding="utf-8") as fp_in:
            chart_figures = [json.loads(line.strip()) for line in fp_in]
        assert len(metadata) == len(vis_code) == len(chart_figures)

        # Load the Vision-language Model (Multimodal LLM)
        vlm_model = VLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_vlm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
        )

        # # Test code: VLM generation
        # cur_messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #             {"type": "text", "text": "Please generate a caption or description of the chart."},
        #         ]
        #     },
        # ]
        # assert os.path.isfile("test_image.png")
        # cur_images = [Image.open("test_image.png")]
        # cur_prompts = vlm_model.processor.apply_chat_template(cur_messages, add_generation_prompt=True)
        # cur_inputs = vlm_model.processor(
        #     text=cur_prompts, images=cur_images, return_tensors="pt").to(vlm_model.model.device)
        # with torch.no_grad():
        #     output_ids = vlm_model.model.generate(**cur_inputs, max_new_tokens=512)
        # output_text = vlm_model.processor.batch_decode(
        #     output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # input_text = vlm_model.processor.batch_decode(
        #     cur_inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # assert len(input_text) == len(cur_prompts) == len(output_text)
        # output_text_pure = []
        # for _input, _prompt, _output in zip(input_text, cur_prompts, output_text):
        #     output_pure = _output[len(_input):]
        #     output_text_pure.append(output_pure)
        #     if self.verbose and self.show_generation:
        #         self.logger.info("================================== >>> output <<<")
        #         self.logger.info(output_pure)
        # self.logger.info(output_text_pure[0].strip())

        chart_captions = []  # List[Dict[str, Any]]
        done_cnt_all, miss_cnt_all = 0, 0
        for metadata_dict, cur_vis_code_dict, cur_chart in zip(metadata, vis_code, chart_figures):
            # Based on the metadata and da_reqs, ask the Code LLM to generate visualization code (Python3 matplotlib).
            cur_chart_cap_dict = dict()
            cur_chart_cap_dict["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            vis_feat_list = cur_vis_code_dict["vis_feat"]
            # code_prompt_list = cur_vis_code_dict["prompts"]
            vis_code_list = cur_vis_code_dict["vis_code"]

            # chart_fig_id_list = cur_chart["fig_id"]
            # chart_fp_list = cur_chart["fig_filepath"]
            chart_fig_base64 = cur_chart["fig_base64"]
            # TODO: future work: consider using code to generate reference captions
            # chart_vis_code_list = cur_chart["vis_code"]
            # chart_code_stat_list = cur_chart["code_stat"]

            assert len(vis_feat_list) == len(vis_code_list) == len(chart_fig_base64) == len(metadata_dict["features"])
            fig_id = 0
            cap_prompt_image_list = []
            for feat_name, cur_vis_code, cur_fig_base64, feat_dict in zip(
                    vis_feat_list, vis_code_list, chart_fig_base64, metadata_dict["features"]):
                fig_id += 1
                if cur_fig_base64 == "":
                    cap_prompt_image_list.append((None, None))
                    continue
                # if self.verbose:
                #     self.logger.info(f">>> >>> Feature: {feat_dict['name']}")
                num_valid, num_unique = feat_dict["num_valid"], feat_dict["num_unique"]
                cur_dtype, numerical_stat = feat_dict["dtype"], feat_dict["numerical_stat"]

                # df_feat = df[feat_dict["name"]]
                # df_feat = df_feat.dropna(axis=0)
                # data_feat = df_feat.tolist()

                cur_cap_prompt = f"""
Dataset Information:
- Dataset Name: {metadata_dict["name"]}
- All Features: {", ".join([x["name"] for x in metadata_dict["features"]])}

Current Feature Information:
- Feature Name: {feat_dict["name"]}
- Data Type: {cur_dtype}
- Number of all rows (feature values): {num_valid}
- Number of unique feature values: {num_unique}
                            """.strip()
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    cur_cap_prompt += "\n" + f"""
- Min of Feature Values: {numerical_stat["min"]:.2f}
- Max of Feature Values: {numerical_stat["max"]:.2f}
- Mean of Feature Values: {numerical_stat["mean"]:.2f}
- Std of Feature Values: {numerical_stat["std"]:.2f}
                                """.strip()

                cur_cap_prompt += "\n\n" + f"""
## Task: Based on the above dataset information (text) and the chart figure (image), \
generate a caption or description of the chart. \
Please be concise and only generate the caption:
                            """.strip()

                cur_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": cur_cap_prompt},
                        ]
                    },
                ]
                # Base64 string to bytes to image
                try:
                    # assert os.path.isfile(cur_chart_fp)
                    # cur_images = [Image.open(cur_chart_fp)]
                    cur_images = [Image.open(BytesIO(base64.b64decode(cur_fig_base64.encode("utf-8"))))]
                    cur_prompts = vlm_model.processor.apply_chat_template(cur_messages, add_generation_prompt=True)
                    cap_prompt_image_list.append((cur_prompts, cur_images))
                except Exception as e:
                    if self.verbose:
                        self.logger.info(e)
                    cap_prompt_image_list.append((None, None))
                    continue

            cur_caption_list = []
            done_cnt, miss_cnt = 0, 0
            for cur_prompts, cur_images in cap_prompt_image_list:
                if cur_prompts is None or cur_images is None:
                    miss_cnt += 1
                    continue

                cur_inputs = vlm_model.processor(
                    text=cur_prompts, images=cur_images, return_tensors="pt").to(vlm_model.model.device)

                with torch.no_grad():
                    output_ids = vlm_model.model.generate(**cur_inputs, max_new_tokens=512)
                output_text = vlm_model.processor.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                input_text = vlm_model.processor.batch_decode(
                    cur_inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                assert len(input_text) == len(cur_prompts) == len(output_text)
                output_text_pure = []
                for _input, _prompt, _output in zip(input_text, cur_prompts, output_text):
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

                cur_caption_list.append(output_text_pure[0].strip())
                done_cnt += 1

            cur_chart_cap_dict["captions"] = cur_caption_list
            chart_captions.append(cur_chart_cap_dict)
            done_cnt_all += done_cnt
            miss_cnt_all += miss_cnt
            if self.debug:
                sys.exit(0)

        # Done all, show statistics
        if self.verbose:
            self.logger.info(f">>> >>> Done All. Statistics: "
                             f"done_cnt_all={done_cnt_all}, miss_cnt_all={miss_cnt_all}")

        # Write the chart captions into jsonl files
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")
        write_cnt = 0
        with open(chart_captions_fp, "w", encoding="utf-8") as fp_out:
            for _item in chart_captions:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {chart_captions_fp}")

        return chart_captions_fp

    def step7_overall_analysis(
            self,
    ) -> str:
        # [Optional] Input all information to Text2Text LLMs and obtain the overall analysis for each table
        overall_analysis = []  # List[str]

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", and "chart_captions.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")

        # Load the Text LLM
        # self.text_llm_model = TextLLM(
        #     verbose=verbose, logger=logger, cuda_dict=cuda_dict,
        #     cache_dir=cache_dir, project_root_dir=project_root_dir,
        #     hf_id=hf_id_text_llm, bsz=bsz, show_generation=show_generation, debug=debug,
        # )

        # Write the data_csv_path and overall_analysis into jsonl files
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        return overall_analysis_fp

    def step8_merge_all_info(
            self,
    ) -> str:
        pass

    def step9_chart_qa_task(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and construct a chart QA evaluation benchmark
        # TODO: manually construct a small evaluation dataset of high quality (or as the guide for text LLMs)
        chart_question = []  # List[str]
        chart_option = []  # List[List[str]] -> five options, one with correct answer (consider distractors' quality)
        chart_answer = []  # List[int] -> each entry: the index of correct answer in each option list

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_figures.jsonl",
        #   "chart_captions.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        # Write each chart QA example into jsonl files
        res_fp = os.path.join(self.data_dir_process, "c4c_qa.jsonl")

        # To upload to Hugging Face datasets

        return res_fp

    def step10_chart_cap_task(
            self,
    ) -> str:
        # Input all information to Text2Text LLMs and construct a chart captioning evaluation benchmark
        # TODO: manually construct a small evaluation dataset of high quality (or as the guide for text LLMs)
        chart_question = []  # List[str]
        chart_option = []  # List[List[str]] -> five options, one with correct answer (consider distractors' quality)
        chart_answer = []  # List[int] -> each entry: the index of correct answer in each option list

        # Get self.datasets_info
        # Load "metadata.jsonl", "da_reqs.jsonl", "vis_code.jsonl", "chart_figures.jsonl",
        #   "chart_captions.jsonl", and "overall_analysis.jsonl"
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")
        vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")

        # Write each chart captioning example into jsonl files
        res_fp = os.path.join(self.data_dir_process, "c4c_cap.jsonl")

        # To upload to Hugging Face datasets

        return res_fp

    def step11_chart_qa_edit_chart(
            self,
    ) -> str:
        # TODO: Do minor modification of the chart figures in our chart QA benchmark for robustness experiments.
        #   Slightly change the chart figures by editing the visualization code.
        pass

    def step12_chart_qa_edit_question(
            self,
    ) -> str:
        # TODO: Do minor modification of the questions in our chart QA benchmark for robustness experiments.
        #   Paraphrase the questions using Text LLMs.
        pass


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
    Run the whole data analysis pipeline (Data-DA-Code-Chart-Caption dataset construction).

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
            c4c_data.step4_vis_code_postprocess()
        case 5:
            c4c_data.step5_exec_vis_code()
        case 6:
            c4c_data.step6_chart_cap_gen()
        case 7:
            c4c_data.step7_overall_analysis()
        case 8:
            c4c_data.step8_merge_all_info()
        case 9:
            c4c_data.step9_chart_qa_task()
        case 10:
            c4c_data.step10_chart_cap_task()
        case 11:
            c4c_data.step11_chart_qa_edit_chart()
        case 11:
            c4c_data.step12_chart_qa_edit_question()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
