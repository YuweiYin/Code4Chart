import os
import sys
import json
import time
from typing import Optional, List, Dict, Any

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
            max_seq_len: int = 1024,
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
            self.logger.info(f">>> Done All. Statistics: "
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
        # Total Running Time: 364.3 sec (6.1 min)
        return chart_figures_fp

    def step6_chart_cap_gen(
            self,
    ) -> str:
        # [Optional] For each chart, we use VLMs to generate the chart captions (descriptions).
        # TODO: future work: the raw data is not provided in the prompt or generated code
        #   since there are too many values in a column,
        #   but maybe it is helpful for the VLMs to perform chart understanding, especially for knowing the numbers

        # Load the metadata, vis_code, and chart figures
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        vis_code_post_fp = os.path.join(self.data_dir_process, "vis_code_post.jsonl")
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(vis_code_post_fp, "r", encoding="utf-8") as fp_in:
            vis_code_post = [json.loads(line.strip()) for line in fp_in]
        with open(chart_figures_fp, "r", encoding="utf-8") as fp_in:
            chart_figures = [json.loads(line.strip()) for line in fp_in]
        assert len(metadata) == len(vis_code_post) == len(chart_figures)

        # Load the Vision-language Model (Multimodal LLM)
        vlm_model = VLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_vlm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
            load_model=True,
        )

        chart_captions = []  # List[Dict[str, Any]]
        done_cnt_all, miss_cnt_all = 0, 0
        for metadata_dict, cur_vis_code_dict, cur_chart in zip(metadata, vis_code_post, chart_figures):
            cur_chart_cap_dict = dict()
            cur_chart_cap_dict["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            # vis_feat_list = cur_vis_code_dict["vis_feat"]
            # code_prompt_list = cur_vis_code_dict["prompts"]
            vis_code_list = cur_vis_code_dict["vis_code"]

            # chart_fig_id_list = cur_chart["fig_id"]
            # chart_fp_list = cur_chart["fig_filepath"]
            chart_fig_base64 = cur_chart["fig_base64"]
            # TODO: future work: consider using code to generate reference captions
            # chart_vis_code_list = cur_chart["vis_code"]
            # chart_code_stat_list = cur_chart["code_stat"]

            assert len(vis_code_list) == len(chart_fig_base64) == len(metadata_dict["features"])
            fig_id = 0
            cap_prompt_image_list = []
            for cur_vis_code, cur_fig_base64, feat_dict in zip(
                    vis_code_list, chart_fig_base64, metadata_dict["features"]):
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
Please be concise and only generate the chart caption:
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
                # assert os.path.isfile(cur_chart_fp)
                # cur_images = [Image.open(cur_chart_fp)]
                cur_images = [Image.open(BytesIO(base64.b64decode(cur_fig_base64.encode("utf-8"))))]
                cur_prompts = vlm_model.processor.apply_chat_template(cur_messages, add_generation_prompt=True)
                if isinstance(cur_prompts, str):
                    cur_prompts = [cur_prompts]
                assert isinstance(cur_prompts, list)
                cap_prompt_image_list.append((cur_prompts, cur_images))

            cur_caption_list = []
            done_cnt, miss_cnt = 0, 0
            for cur_prompts, cur_images in cap_prompt_image_list:
                if cur_prompts is None or cur_images is None:
                    cur_caption_list.append(None)  # Can NOT be "" since json.dumps will ignore it
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

                # if self.debug:
                #     self.logger.info(f">>> len(cur_prompts) = {len(cur_prompts)}")
                #     self.logger.info(f">>> len(input_text) = {len(input_text)}")
                #     self.logger.info(f">>> len(output_text) = {len(output_text)}")
                #     self.logger.info(f">>> cur_prompts = {cur_prompts}")
                #     self.logger.info(f">>> input_text = {input_text}")
                #     self.logger.info(f">>> output_text = {output_text}")

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
            if self.verbose:
                self.logger.info(f">>> >>> Done [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}. "
                                 f"done_cnt={done_cnt}, miss_cnt={miss_cnt}")
            if self.debug:
                sys.exit(0)

        # Done all, show statistics
        if self.verbose:
            self.logger.info(f">>> Done All. Statistics: "
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
        # Total Running Time: 949.6 sec (15.8 min)
        return chart_captions_fp

    def step7_overall_analysis(
            self,
    ) -> str:
        # [Optional] Input all information to Text2Text LLMs and obtain the overall analysis for each table
        # TODO: Overall analysis and insights

        # Load the metadata, vis_code, and chart figures
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")
        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(chart_captions_fp, "r", encoding="utf-8") as fp_in:
            chart_captions = [json.loads(line.strip()) for line in fp_in]
        assert len(metadata) == len(chart_captions)

        # Load the Text LLM
        text_llm = TextLLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_text_llm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
        )

        overall_analysis = []  # List[Dict[str, Any]]
        for metadata_dict, cur_chart_cap_dict in zip(metadata, chart_captions):
            # Based on all information we have, ask Text LLMs to generate the overall analysis for each table
            cur_analysis_dict = dict()
            cur_analysis_dict["id"] = metadata_dict["id"]
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            cur_caption_list = cur_chart_cap_dict["captions"]

            cur_analysis_prompt = f"""
## Dataset Information:
- Dataset name: {metadata_dict["name"]}
- Dataset Description: {metadata_dict["description"]}

## Dataset Statistics:
- Dataset size (the number of rows): {metadata_dict["num_row"]}
- The number of columns (features): {metadata_dict["num_col"]}
- All feature names: {", ".join([x["name"] for x in metadata_dict["features"]])}
- All feature data types: {", ".join([x["dtype"] for x in metadata_dict["features"]])}

## Chart Captions:
            """.strip()

            for cur_caption_text, feat_dict in zip(cur_caption_list, metadata_dict["features"]):
                if isinstance(cur_caption_text, str) and len(cur_caption_text) > 0:
                    cur_caption_text = cur_caption_text.replace("\n", " ").strip()
                    cur_analysis_prompt += "\n" + f"""
### Chart Caption of Feature \"{feat_dict["name"]}\" (data type: {feat_dict["dtype"]}):
{cur_caption_text}
                    """.strip()

            cur_analysis_prompt += "\n\n" + f"""
## Task: Based on the above dataset information and the chart captions (dataset visualization), \
generate an conclusive overall analysis for this dataset. \
Please be concise and only generate the conclusion:
            """.strip()

            gen_dict = text_llm.run_generation(
                prompts=[cur_analysis_prompt], model=text_llm.model, tokenizer=text_llm.tokenizer_gen,
                need_tokenize=True, max_new_tokens=1024,
                temperature=0.1, top_p=0.1,  # Be more deterministic when choosing an option
            )
            output_text = gen_dict["output_text"][0].strip()

            cur_analysis_dict["analysis_prompt"] = cur_analysis_prompt
            cur_analysis_dict["overall_analysis"] = output_text
            overall_analysis.append(cur_analysis_dict)
            if self.debug:
                sys.exit(0)

        # Write the overall_analysis into jsonl files
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")
        write_cnt = 0
        with open(overall_analysis_fp, "w", encoding="utf-8") as fp_out:
            for _item in overall_analysis:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {overall_analysis_fp}")
        # Total Running Time: 593.4 sec (9.9 min)
        return overall_analysis_fp

    def step8_merge_all_info(
            self,
    ) -> str:
        metadata_fp = os.path.join(self.data_dir_process, "metadata.jsonl")  # Step 1
        da_reqs_fp = os.path.join(self.data_dir_process, "da_reqs.jsonl")  # Step 2
        # vis_code_fp = os.path.join(self.data_dir_process, "vis_code.jsonl")  # Step 3
        vis_code_post_fp = os.path.join(self.data_dir_process, "vis_code_post.jsonl")  # Step 4
        chart_figures_fp = os.path.join(self.data_dir_process, "chart_figures.jsonl")  # Step 5
        chart_captions_fp = os.path.join(self.data_dir_process, "chart_captions.jsonl")  # Step 6
        overall_analysis_fp = os.path.join(self.data_dir_process, "overall_analysis.jsonl")  # Step 7

        with open(metadata_fp, "r", encoding="utf-8") as fp_in:
            metadata = [json.loads(line.strip()) for line in fp_in]
        with open(da_reqs_fp, "r", encoding="utf-8") as fp_in:
            da_reqs = [json.loads(line.strip()) for line in fp_in]
        # with open(vis_code_fp, "r", encoding="utf-8") as fp_in:
        #     vis_code = [json.loads(line.strip()) for line in fp_in]
        with open(vis_code_post_fp, "r", encoding="utf-8") as fp_in:
            vis_code_post = [json.loads(line.strip()) for line in fp_in]
        with open(chart_figures_fp, "r", encoding="utf-8") as fp_in:
            chart_figures = [json.loads(line.strip()) for line in fp_in]
        with open(chart_captions_fp, "r", encoding="utf-8") as fp_in:
            chart_captions = [json.loads(line.strip()) for line in fp_in]
        with open(overall_analysis_fp, "r", encoding="utf-8") as fp_in:
            overall_analysis = [json.loads(line.strip()) for line in fp_in]
        assert (len(metadata) == len(da_reqs) == len(vis_code_post) ==
                len(chart_figures) == len(chart_captions) == len(overall_analysis))

        all_info = []  # List[Dict[str, Any]]
        for metadata_dict, da_reqs_dict, vis_code_dict, chart_fig_dict, chart_cap_dict, analysis_dict in zip(
                metadata, da_reqs, vis_code_post, chart_figures, chart_captions, overall_analysis):
            cur_info_dict = dict()
            if self.verbose:
                self.logger.info(f">>> [id={metadata_dict['id']}] Dataset: {metadata_dict['name']}")

            cur_info_dict["id"] = metadata_dict["id"]
            cur_info_dict["url"] = metadata_dict["url"]
            cur_info_dict["name"] = metadata_dict["name"]
            cur_info_dict["description"] = metadata_dict["description"]
            cur_info_dict["filename"] = metadata_dict["filename"]
            cur_info_dict["filepath"] = metadata_dict["filepath"]
            cur_info_dict["num_row"] = metadata_dict["num_row"]
            cur_info_dict["num_col"] = metadata_dict["num_col"]
            cur_info_dict["features"] = metadata_dict["features"]

            cur_info_dict["da_reqs"] = da_reqs_dict["da_reqs"]
            # cur_info_dict["da_reqs_prompts"] = da_reqs_dict["prompts"]

            cur_info_dict["vis_code_raw"] = vis_code_dict["vis_code"]
            cur_info_dict["vis_code_features"] = vis_code_dict["vis_feat"]
            # cur_info_dict["vis_code_prompts"] = vis_code_dict["prompts"]
            cur_info_dict["vis_code_filepath"] = vis_code_dict["code_filepath"]

            cur_info_dict["vis_code_clean"] = chart_fig_dict["vis_code"]
            cur_info_dict["vis_code_stat"] = chart_fig_dict["code_stat"]
            cur_info_dict["chart_figure_base64"] = chart_fig_dict["fig_base64"]
            cur_info_dict["chart_figure_filepath"] = chart_fig_dict["fig_filepath"]

            cur_info_dict["captions"] = chart_cap_dict["captions"]

            cur_info_dict["analysis"] = analysis_dict["overall_analysis"]
            # cur_info_dict["analysis_prompts"] = analysis_dict["analysis_prompt"]

            all_info.append(cur_info_dict)

        # Write all the info into jsonl files
        all_info_fp = os.path.join(self.data_dir_process, "all_info.json")
        with open(all_info_fp, "w", encoding="utf-8") as fp_out:
            json.dump(all_info, fp_out, cls=NumpyEncoder, indent=4)

        all_info_fp = os.path.join(self.data_dir_process, "all_info.jsonl")
        write_cnt = 0
        with open(all_info_fp, "w", encoding="utf-8") as fp_out:
            for _item in all_info:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {all_info_fp}")
        return all_info_fp

    def step9_chart_qa_task(
            self,
    ) -> str:
        # Construct a chart QA evaluation benchmark by feeding all information to Text LLMs.
        # After generation, manually post-process the generated questions, options, and answers (quality control).

        # Load processed all_info data
        all_info_fp = os.path.join(self.data_dir_process, "all_info.jsonl")
        with open(all_info_fp, "r", encoding="utf-8") as fp_in:
            all_info = [json.loads(line.strip()) for line in fp_in]

        # Load the Text LLM
        text_llm = TextLLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_text_llm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
        )

        c4c_chart_qa = []  # List[Dict[str, Any]]
        for cur_info_dict in all_info:
            if self.verbose:
                self.logger.info(f">>> [id={cur_info_dict['id']}] Dataset: {cur_info_dict['name']}")

            cur_qa_dict = cur_info_dict

            assert len(cur_qa_dict["features"]) == len(cur_qa_dict["captions"])

            prompt_list = []
            for feat_dict, cur_caption_text in zip(cur_qa_dict["features"], cur_qa_dict["captions"]):
                # Here, we construct chart QA task's questions, options, and answers mainly based on chart captions
                # if self.verbose:
                #     self.logger.info(f">>> >>> Feature: {feat_dict['name']}")
                # Dataset Information:
                # - Dataset Name: {cur_qa_dict["name"]}
                # - All Features: {", ".join([x["name"] for x in cur_qa_dict["features"]])}
                qa_prompt = f"""
Feature Information of Dataset {cur_qa_dict["name"]}:
- Feature Name: {feat_dict["name"]}
- Data Type: {feat_dict["dtype"]}
- Number of all rows (feature values): {feat_dict["num_valid"]}
- Number of unique feature values: {feat_dict["num_unique"]}
                """.strip()
                numerical_stat = feat_dict["numerical_stat"]
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    qa_prompt += "\n" + f"""
- Min of Feature Values: {numerical_stat["min"]:.2f}
- Max of Feature Values: {numerical_stat["max"]:.2f}
- Mean of Feature Values: {numerical_stat["mean"]:.2f}
- Std of Feature Values: {numerical_stat["std"]:.2f}
                    """.strip()
                if isinstance(cur_caption_text, str) and len(cur_caption_text) > 0:
                    cur_caption_text = cur_caption_text.replace("\n", " ").strip()
                    qa_prompt += "\n\n" + f"""
Chart Caption of Feature \"{feat_dict["name"]}\" (data type: {feat_dict["dtype"]}):
{cur_caption_text}
                    """.strip()

                qa_prompt += "\n\n" + f"""
## Task: Based on the dataset feature information and the corresponding chart captions above,
construct one chart question-answering test, including a question, five options, and a correct answer. \
Please be concise and only generate the question, options, and answer:
                """.strip()

                prompt_list.append(qa_prompt)

            cur_qa_dict["chart_qa"] = []
            for prompt in prompt_list:
                gen_dict = text_llm.run_generation(
                    prompts=[prompt], model=text_llm.model, tokenizer=text_llm.tokenizer_gen,
                    need_tokenize=True, max_new_tokens=512,
                    temperature=0.1, top_p=0.1,  # Be more deterministic when choosing an option
                )
                output_text = gen_dict["output_text"][0].strip()
                cur_qa_dict["chart_qa"].append(output_text)
                # Post-processing: manually extract questions, options, and answers from the generated text.
                #   Also, consider the balance of answers (five options, random guess accuracy ~= 20%)
                # cur_qa_dict["chart_qa"] = {
                #     "question": "",
                #     "options": {
                #         "A": "",
                #         "B": "",
                #         "C": "",
                #         "D": "",
                #         "E": "",
                #     },  # Five options, one of which is correct (consider the quality of the distractors)
                #     "answer": "",
                # }

            c4c_chart_qa.append(cur_qa_dict)
            if self.debug:
                sys.exit(0)

        # [Later] To upload to Hugging Face datasets

        # Write the chart QA benchmark into jsonl files
        c4c_chart_qa_fp = os.path.join(self.data_dir_process, "c4c_chart_qa.json")
        with open(c4c_chart_qa_fp, "w", encoding="utf-8") as fp_out:
            json.dump(c4c_chart_qa, fp_out, cls=NumpyEncoder, indent=4)

        c4c_chart_qa_fp = os.path.join(self.data_dir_process, "c4c_chart_qa.jsonl")
        write_cnt = 0
        with open(c4c_chart_qa_fp, "w", encoding="utf-8") as fp_out:
            for _item in c4c_chart_qa:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {c4c_chart_qa_fp}")
        # Total Running Time: 2658.4 sec (44.3 min)
        return c4c_chart_qa_fp

    def step10_chart_qa_edit_chart(
            self,
    ) -> str:
        # Do minor modification of the chart figures in our chart QA benchmark for robustness experiments.
        # Slightly change the chart figures by editing the visualization code.

        # Step 1: Pick some figures that the model can correctly classify in the base settings.
        pick_dataset = ["1", "2", "4", "5", "7", "8", "9"]
        pick_dataset_set = set(pick_dataset)
        pick_index_list = [9, 2, 8, 6, 2, 3, 12]
        assert len(pick_dataset) == len(pick_index_list)

        # Load the original chart QA benchmark
        c4c_chart_qa_fp = os.path.join(self.data_dir_process, "c4c_chart_qa_post.json")
        with open(c4c_chart_qa_fp, "r", encoding="utf-8") as fp_in:
            # c4c_chart_qa = [json.loads(line.strip()) for line in fp_in]
            c4c_chart_qa = json.load(fp_in)

        c4c_chart_qa_pick = []
        done_cnt = 0
        pick_i = -1
        for cur_qa_dict in c4c_chart_qa:
            if cur_qa_dict["id"] not in pick_dataset_set:
                continue
            pick_i += 1
            cur_pick_index = pick_index_list[pick_i]
            if self.verbose:
                self.logger.info(f">>> [id={cur_qa_dict['id']}] Dataset: {cur_qa_dict['name']}; "
                                 f"cur_pick_index = {cur_pick_index}")

            cur_c4c_chart_qa_pick = dict()
            cur_c4c_chart_qa_pick["id"] = cur_qa_dict["id"]
            cur_c4c_chart_qa_pick["url"] = cur_qa_dict["url"]
            cur_c4c_chart_qa_pick["name"] = cur_qa_dict["name"]
            cur_c4c_chart_qa_pick["description"] = cur_qa_dict["description"]
            cur_c4c_chart_qa_pick["filename"] = cur_qa_dict["filename"]
            cur_c4c_chart_qa_pick["filepath"] = cur_qa_dict["filepath"]
            cur_c4c_chart_qa_pick["num_row"] = cur_qa_dict["num_row"]
            cur_c4c_chart_qa_pick["num_col"] = cur_qa_dict["num_col"]
            cur_c4c_chart_qa_pick["analysis"] = cur_qa_dict["analysis"]
            cur_c4c_chart_qa_pick["features"] = cur_qa_dict["features"][cur_pick_index]
            cur_c4c_chart_qa_pick["da_reqs"] = cur_qa_dict["da_reqs"][cur_pick_index]
            cur_c4c_chart_qa_pick["vis_code_raw"] = cur_qa_dict["vis_code_raw"][cur_pick_index]
            cur_c4c_chart_qa_pick["vis_code_features"] = cur_qa_dict["vis_code_features"][cur_pick_index]
            cur_c4c_chart_qa_pick["vis_code_filepath"] = cur_qa_dict["vis_code_filepath"][cur_pick_index]
            cur_c4c_chart_qa_pick["vis_code_clean"] = cur_qa_dict["vis_code_clean"][cur_pick_index]
            cur_c4c_chart_qa_pick["vis_code_stat"] = cur_qa_dict["vis_code_stat"][cur_pick_index]
            cur_c4c_chart_qa_pick["chart_figure_base64"] = cur_qa_dict["chart_figure_base64"][cur_pick_index]
            cur_c4c_chart_qa_pick["chart_figure_filepath"] = cur_qa_dict["chart_figure_filepath"][cur_pick_index]
            cur_c4c_chart_qa_pick["captions"] = cur_qa_dict["captions"][cur_pick_index]
            cur_c4c_chart_qa_pick["chart_qa"] = cur_qa_dict["chart_qa"][cur_pick_index]
            cur_c4c_chart_qa_pick["chart_qa_clean"] = cur_qa_dict["chart_qa_clean"][cur_pick_index]

            # Step 2: Change bar color (the default color "C1") in VisCode:
            #   color="blue", color="green", color="red", color="cyan", color="magenta",
            #   color="yellow", color="orange", color="black"  # (color="white", edgecolor="black")
            # TODO: future work: Other changes, e.g., figure background color, legend location, and figure orientation.
            color_list = ["blue", "cyan", "green", "red", "magenta", "orange", "yellow", "black"]

            original_vis_code = cur_c4c_chart_qa_pick["vis_code_clean"]
            original_chart_base64 = cur_c4c_chart_qa_pick["chart_figure_base64"]
            cur_c4c_chart_qa_pick["vis_code_edit_bar_color"] = {
                "original": original_vis_code,  # "C0" color
            }
            cur_c4c_chart_qa_pick["chart_figure_edit_bar_color"] = {
                "original": original_chart_base64,  # "C0" color
            }
            try:
                exec(original_vis_code)
                done_cnt += 1
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> >>> Exception: {e} --- Error exec the original file")
                raise ValueError(f">>> Error exec session.")

            for color in color_list:
                assert "plt.savefig(" in original_vis_code
                assert "plt.bar(" in original_vis_code or "plt.hist(" in original_vis_code
                original_vis_code_lines = original_vis_code.split("\n")
                edit_vis_code_lines = []
                new_save_fp = None
                for cur_line in original_vis_code_lines:
                    cur_line = cur_line.rstrip()
                    if "plt.savefig(" in cur_line:  # change the saving filepath
                        assert "process/chart_figure/" in cur_line
                        new_line = cur_line.replace("process/chart_figure/", "process/chart_figure_pick/")
                        assert ".png" in new_line
                        new_line = new_line.replace(".png", f"-{color}.png")
                        new_save_fp = new_line.split("plt.savefig(")[-1].split(",")[0].strip()
                        if new_save_fp[0] == "\'" or new_save_fp[0] == "\"":  # Ignore the left-most quotation mark
                            new_save_fp = new_save_fp[1:]
                        if new_save_fp[-1] == "\'" or new_save_fp[-1] == "\"":  # Ignore the right-most quotation mark
                            new_save_fp = new_save_fp[:-1]
                    elif "plt.bar(" in cur_line:  # Specify bar color
                        if "edgecolor=" in cur_line:
                            new_line = cur_line.split("edgecolor=")[0].rstrip()
                            if "color=" in new_line:
                                new_line = new_line.split("color=")[0].rstrip()
                            if new_line[-1] == ",":
                                new_line += f" color='{color}')"
                            else:
                                new_line += f", color='{color}')"
                        else:
                            assert cur_line[-1] == ")"
                            new_line = cur_line[:-1] + f", color='{color}')"
                    elif "plt.hist(" in cur_line:  # Specify bar color of histogram
                        if "edgecolor=" in cur_line:
                            new_line = cur_line.split("edgecolor=")[0].rstrip()
                            if "color=" in new_line:
                                new_line = new_line.split("color=")[0].rstrip()
                            if new_line[-1] == ",":
                                new_line += f" color='{color}', edgecolor='black')"
                            else:
                                new_line += f", color='{color}', edgecolor='black')"
                        else:
                            assert cur_line[-1] == ")"
                            new_line = cur_line[:-1] + f", color='{color}', edgecolor='black')"
                    else:
                        new_line = cur_line
                    edit_vis_code_lines.append(new_line)

                assert new_save_fp is not None
                edit_vis_code = "\n".join(edit_vis_code_lines)
                cur_c4c_chart_qa_pick["vis_code_edit_bar_color"][color] = edit_vis_code

                # Step 3: Execute the edited VisCode and encode the saved chart figures
                try:
                    exec(edit_vis_code)
                    assert os.path.isfile(new_save_fp)

                    # Base64 encoding
                    with open(new_save_fp, "rb") as img_fp_in:
                        img_base64 = base64.b64encode(img_fp_in.read())
                    img_base64_str = img_base64.decode("utf-8")
                    cur_c4c_chart_qa_pick["chart_figure_edit_bar_color"][color] = img_base64_str

                    done_cnt += 1
                except Exception as e:
                    if self.verbose:
                        self.logger.info(f">>> >>> Exception: {e} --- Error exec file: {new_save_fp}")
                    raise ValueError(f">>> Error exec session.")

            assert (len(cur_c4c_chart_qa_pick["vis_code_edit_bar_color"]) ==
                    len(cur_c4c_chart_qa_pick["chart_figure_edit_bar_color"]))
            c4c_chart_qa_pick.append(cur_c4c_chart_qa_pick)

        # [Later] To upload to Hugging Face datasets

        # Write the chart QA benchmark into jsonl files
        c4c_chart_qa_post_edit_fp = os.path.join(self.data_dir_process, "c4c_chart_qa_post_edit.json")
        with open(c4c_chart_qa_post_edit_fp, "w", encoding="utf-8") as fp_out:
            json.dump(c4c_chart_qa_pick, fp_out, cls=NumpyEncoder, indent=4)

        c4c_chart_qa_post_edit_fp = os.path.join(self.data_dir_process, "c4c_chart_qa_post_edit.jsonl")
        write_cnt = 0
        with open(c4c_chart_qa_post_edit_fp, "w", encoding="utf-8") as fp_out:
            for _item in c4c_chart_qa_pick:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> done_cnt = {done_cnt}; "
                             f"write_cnt = {write_cnt} to file: {c4c_chart_qa_post_edit_fp}")
        # Total Running Time: 481.2 sec (8.0 min)
        return c4c_chart_qa_post_edit_fp

    # def step11_chart_qa_edit_question(
    #         self,
    # ) -> str:
    #     # TODO: future work:
    #     #   Do minor modification of the questions in our chart QA benchmark for robustness experiments.
    #     #   Paraphrase the questions using Text LLMs.
    #     pass

    # def step12_chart_cap_task(
    #         self,
    # ) -> str:
    #     # [Optional] Construct a chart captioning evaluation benchmark by feeding all information to Text LLMs.
    #     pass


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
            c4c_data.step10_chart_qa_edit_chart()
        # case 11:
        #     c4c_data.step11_chart_qa_edit_question()
        # case 12:
        #     c4c_data.step12_chart_cap_task()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
