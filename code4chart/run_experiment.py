import os
import sys
import json
import time
from typing import Optional

import fire
import numpy as np

import base64
from PIL import Image
from io import BytesIO

import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login as hf_login
# from datasets import load_dataset, DatasetDict, Dataset

# from code4chart.text_llm import TextLLM
# from code4chart.code_llm import CodeLLM
from code4chart.vlm import VLM
# from code4chart.default_inputs import DefaultInputs
from code4chart.init_functions import logger_setup, cuda_setup, random_setup
from code4chart.numpy_encoder import NumpyEncoder


class Code4ChartExp:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
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
        :param cache_dir: The root directory of the cache.
        :param project_root_dir: The directory of the project root.
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
        self.hf_id_vlm = hf_id_vlm

        # Data and checkpoint directory
        self.data_dir = os.path.join(project_root_dir, "data/code4chart")
        self.ckpt_dir = os.path.join(project_root_dir, "ckpt/code4chart")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.data_dir_raw = os.path.join(self.data_dir, "raw")
        self.data_dir_process = os.path.join(self.data_dir, "process")
        self.data_dir_results = os.path.join(self.data_dir, "results", self.hf_id_vlm.split("/")[-1])
        os.makedirs(self.data_dir_raw, exist_ok=True)
        os.makedirs(self.data_dir_process, exist_ok=True)
        os.makedirs(self.data_dir_results, exist_ok=True)

    def run_chart_qa(
            self,
            add_ds_info: bool = False,
            add_code: bool = False,
            remove_comments: bool = False,
            use_cot: bool = False,
            few_shot: int = 0,
    ) -> str:
        # Load our chart QA benchmark
        c4c_chart_qa_fp = os.path.join(self.data_dir_process, "c4c_chart_qa_post.json")
        with open(c4c_chart_qa_fp, "r", encoding="utf-8") as fp_in:
            # c4c_chart_qa = [json.loads(line.strip()) for line in fp_in]
            c4c_chart_qa = json.load(fp_in)

        # Correct choices distribution statistics
        # all_ans = [0, 0, 0, 0, 0]
        # all_ans_valid = [0, 0, 0, 0, 0]  # {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        # choice2index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        # for ds_meta in c4c_chart_qa:
        #     chart_figure_base64 = ds_meta["chart_figure_base64"]
        #     chart_qa_clean = ds_meta["chart_qa_clean"]
        #     assert len(chart_figure_base64) == len(chart_qa_clean)
        #     for cur_fig, cur_qa in zip(chart_figure_base64, chart_qa_clean):
        #         cur_ans = cur_qa["answer"]
        #         assert cur_ans in choice2index and isinstance(cur_fig, str)
        #         all_ans[choice2index[cur_ans]] += 1
        #         if cur_fig != "":
        #             all_ans_valid[choice2index[cur_ans]] += 1

        # Load the Vision-language Model (Multimodal LLM)
        # TODO: future work: test VLMs that fine-tuned on [our] chart datasets, expecting performance improvement
        #   Also, test proprietary LLMs like GPT-4
        vlm_model = VLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_vlm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
            load_model=True,
        )

        all_qa_results = []  # List[Dict[str, Any]]
        done_cnt_all, miss_cnt_all = 0, 0
        fail_to_answer_cnt_all = 0
        for cur_qa_dict in c4c_chart_qa:
            # cur_qa_results = dict()
            cur_qa_results = []
            if self.verbose:
                self.logger.info(f">>> [id={cur_qa_dict['id']}] Dataset: {cur_qa_dict['name']}")

            # cur_qa_dict["id"] = cur_info_dict["id"]
            # cur_qa_dict["url"] = cur_info_dict["url"]
            # cur_qa_dict["name"] = cur_info_dict["name"]
            # cur_qa_dict["description"] = cur_info_dict["description"]
            # cur_qa_dict["filename"] = cur_info_dict["filename"]
            # cur_qa_dict["filepath"] = cur_info_dict["filepath"]
            # cur_qa_dict["num_row"] = cur_info_dict["num_row"]
            # cur_qa_dict["num_col"] = cur_info_dict["num_col"]
            # cur_qa_dict["features"] = cur_info_dict["features"]
            #
            # cur_qa_dict["da_reqs"] = cur_info_dict["da_reqs"]
            # cur_qa_dict["captions"] = cur_info_dict["captions"]
            # cur_qa_dict["analysis"] = cur_info_dict["analysis"]
            #
            # # cur_qa_dict["vis_code_raw"] = cur_info_dict["vis_code_raw"]
            # cur_qa_dict["vis_code_features"] = cur_info_dict["vis_code_features"]
            # cur_qa_dict["vis_code_filepath"] = cur_info_dict["vis_code_filepath"]
            #
            # cur_qa_dict["vis_code_clean"] = cur_info_dict["vis_code_clean"]
            # cur_qa_dict["vis_code_stat"] = cur_info_dict["vis_code_stat"]
            # cur_qa_dict["chart_figure_base64"] = cur_info_dict["chart_figure_base64"]
            # cur_qa_dict["chart_figure_filepath"] = cur_info_dict["chart_figure_filepath"]

            chart_figure_base64 = cur_qa_dict["chart_figure_base64"]
            # chart_qa = cur_qa_dict["chart_qa"]
            chart_qa = cur_qa_dict["chart_qa_clean"]
            vis_code = cur_qa_dict["vis_code_clean"]
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
            # cur_qa_results["chart_figure_base64"] = chart_figure_base64
            # cur_qa_results["chart_qa"] = chart_qa
            # cur_qa_results["vis_code"] = vis_code

            choice_set = {"A", "B", "C", "D", "E"}

            assert len(chart_qa) == len(chart_figure_base64) == len(vis_code) == len(cur_qa_dict["features"])
            qa_prompt_image_list = []
            cur_questions, cur_options, cur_answers = [], [], []
            for cur_chart_qa, cur_fig_base64, cur_vis_code, feat_dict in zip(
                    chart_qa, chart_figure_base64, vis_code, cur_qa_dict["features"]):
                if cur_fig_base64 == "":
                    cur_questions.append(None)
                    cur_options.append(None)
                    cur_answers.append(None)
                    qa_prompt_image_list.append((None, None))
                    continue

                question, options, answer = cur_chart_qa["question"], cur_chart_qa["options"], cur_chart_qa["answer"]
                cur_questions.append(question)
                cur_options.append(options)
                cur_answers.append(answer)

                # The dataset/feature information
                cur_ds_info_prompt = f"""
Dataset Information:
- Dataset Name: {cur_qa_dict["name"]}
- All Features: {", ".join([x["name"] for x in cur_qa_dict["features"]])}

Current Feature Information:
- Feature Name: {feat_dict["name"]}
- Data Type: {feat_dict["dtype"]}
- Number of all rows (feature values): {feat_dict["num_valid"]}
- Number of unique feature values: {feat_dict["num_unique"]}
                """.strip()

                numerical_stat = feat_dict["numerical_stat"]
                if isinstance(numerical_stat, dict) and len(numerical_stat) > 0:
                    cur_ds_info_prompt += "\n" + f"""
- Min of Feature Values: {numerical_stat["min"]:.2f}
- Max of Feature Values: {numerical_stat["max"]:.2f}
- Mean of Feature Values: {numerical_stat["mean"]:.2f}
- Std of Feature Values: {numerical_stat["std"]:.2f}
                    """.strip()

                cur_vis_code = cur_vis_code.strip()
                if self.data_dir in cur_vis_code:
                    cur_vis_code = cur_vis_code.replace(self.data_dir, "data").strip()
                # Optionally add the dataset/feature information
                # TODO: when generating the chart QA benchmark, the model see the dataset information,
                #   but letting the test model too see this information may result in answer leaking to some extent.
                if add_ds_info:
                    # Optionally add the visualization code
                    if add_code:
                        cur_qa_prompt = cur_ds_info_prompt + "\n\n" + f"""
The Python code to generate the chart is as follows:
```python
{cur_vis_code.strip()}
```

## Task: Based on the above information (text and code) and the chart figure (image), \
answer the following question by choosing an option from "A", "B", "C", "D", and "E". \
Be concise, give a short answer, and only output your choice.
                        """.strip()
                    else:
                        cur_qa_prompt = cur_ds_info_prompt + "\n\n" + f"""
## Task: Based on the above information (text) and the chart figure (image), \
answer the following question by choosing an option from "A", "B", "C", "D", and "E". \
Be concise, give a short answer, and only output your choice.
                        """.strip()
                else:  # No dataset/feature information
                    if add_code:
                        cur_qa_prompt = f"""
The Python code to generate the chart is as follows:
```python
{cur_vis_code.strip()}
```

## Task: Based on the above information (code) and the chart figure (image), \
answer the following question by choosing an option from "A", "B", "C", "D", and "E". \
Be concise, give a short answer, and only output your choice.
                        """.strip()
                    else:
                        cur_qa_prompt = f"""
## Task: Based on the chart figure (image), \
answer the following question by choosing an option from "A", "B", "C", "D", and "E". \
Be concise, give a short answer, and only output your choice.
                        """.strip()

                # Add question and options
                cur_qa_prompt += "\n" + f"""
Question: {question}
A: {options["A"]}
B: {options["B"]}
C: {options["C"]}
D: {options["D"]}
E: {options["E"]}
Answer:
                """.strip()

                # Chain-of-thought prompting
                if use_cot:
                    cur_qa_prompt += " Let's think step by step."

                cur_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": cur_qa_prompt},
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
                qa_prompt_image_list.append((cur_prompts, cur_images))

            cur_results, cur_choices = [], []
            done_cnt, miss_cnt = 0, 0
            fail_to_answer_cnt = 0
            for cur_prompts, cur_images in qa_prompt_image_list:
                if cur_prompts is None or cur_images is None:
                    cur_results.append(None)  # Can NOT be "" since json.dumps will ignore it
                    cur_choices.append(None)
                    miss_cnt += 1
                    continue

                cur_inputs = vlm_model.processor(
                    text=cur_prompts, images=cur_images, return_tensors="pt").to(vlm_model.model.device)

                with torch.no_grad():
                    output_ids = vlm_model.model.generate(**cur_inputs, max_new_tokens=200)
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

                # Pick VLMs' choice
                cur_output = output_text_pure[0].strip()
                cur_results.append(cur_output)
                assert isinstance(cur_output, str)
                if cur_output.startswith("(") or cur_output.startswith("\"") or cur_output.startswith("\'"):
                    cur_output = cur_output[1:]
                if cur_output.endswith(".") or cur_output.endswith(")") or \
                        cur_output.endswith("\"") or cur_output.endswith("\'"):
                    cur_output = cur_output[:-1]
                if cur_output[-1] in choice_set:  # {"A", "B", "C", "D", "E"}
                    cur_choices.append(cur_output[-1])
                elif cur_output[0] in choice_set:  # {"A", "B", "C", "D", "E"}
                    if cur_output.startswith("Base") or cur_output.startswith("Accord"):
                        cur_choices.append("X")
                    else:
                        cur_choices.append(cur_output[0])
                else:
                    fail_to_answer_cnt += 1
                    cur_choices.append("X")
                    # TODO: The VLM often does not output A/B/C/D/E directly, so we need to read the generated output;
                    #   Or, choose an A/B/C/D/E option by observing the first logits (vocab prob).
                done_cnt += 1

            assert len(cur_results) == len(cur_choices) == len(chart_qa) == len(chart_figure_base64)
            assert len(cur_questions) == len(cur_options) == len(cur_answers) == len(cur_results)
            for _q, _o, _a, _r, _c in zip(cur_questions, cur_options, cur_answers, cur_results, cur_choices):
                cur_qa_results.append({
                    "question": _q,
                    "options": _o,
                    "answer": _a,
                    "model_output": _r,
                    "model_choice": _c,
                })
            # cur_qa_results["model_results"] = cur_results
            # cur_qa_results["model_choices"] = cur_choices
            # cur_qa_results["questions"] = cur_questions
            # cur_qa_results["options"] = cur_options
            # cur_qa_results["answers"] = cur_answers
            # all_qa_results.append(cur_qa_results)
            all_qa_results.extend(cur_qa_results)
            done_cnt_all += done_cnt
            miss_cnt_all += miss_cnt
            fail_to_answer_cnt_all += fail_to_answer_cnt
            if self.verbose:
                self.logger.info(f">>> >>> Done [id={cur_qa_dict['id']}] Dataset: {cur_qa_dict['name']}. "
                                 f"done_cnt={done_cnt}, miss_cnt={miss_cnt}, fail_to_answer_cnt={fail_to_answer_cnt}")
            if self.debug:
                sys.exit(0)

        # Done all, show statistics
        if self.verbose:
            self.logger.info(f">>> Done All. Statistics: "
                             f"done_cnt_all={done_cnt_all}, miss_cnt_all={miss_cnt_all}, "
                             f"fail_to_answer_cnt_all={fail_to_answer_cnt_all}")

        # Compute the chart QA accuracy. TODO: Needs post-processing to determine the model's choice
        all_answers, all_choices = [], []
        for cur_qa_res in all_qa_results:
            all_answers.append(cur_qa_res["answer"])
            all_choices.append(cur_qa_res["model_choice"])
        all_answers = [_item.strip() for _item in all_answers if isinstance(_item, str)]
        all_choices = [_item.strip() for _item in all_choices if isinstance(_item, str)]
        try:
            assert len(all_answers) == len(all_choices)
            all_answers_np = np.array(all_answers)
            all_choices_np = np.array(all_choices)
            acc = np.mean(all_answers_np == all_choices_np).item()
            if self.verbose:
                self.logger.info(f">>> [#item = {len(all_choices)}] Accuracy: {acc:.5f}")
        except Exception as e:
            if self.verbose:
                self.logger.info(e)

        # Write the QA and results into jsonl files
        add_code_tag = "1" if add_code else "0"
        add_ds_info_tag = "1" if add_ds_info else "0"
        use_cot_tag = "1" if use_cot else "0"
        remove_comments_tag = "1" if remove_comments else "0"
        all_qa_results_fp = os.path.join(
            self.data_dir_results,
            f"all_qa_results-{add_code_tag}_{add_ds_info_tag}_{use_cot_tag}_{remove_comments_tag}_{few_shot}.json")
        with open(all_qa_results_fp, "w", encoding="utf-8") as fp_out:
            json.dump(all_qa_results, fp_out, cls=NumpyEncoder, indent=4)

        all_qa_results_fp += "l"  # ".jsonl"
        write_cnt = 0
        with open(all_qa_results_fp, "w", encoding="utf-8") as fp_out:
            for _item in all_qa_results:
                fp_out.write(json.dumps(_item, cls=NumpyEncoder) + "\n")
                write_cnt += 1

        if self.verbose:
            self.logger.info(f">>> write_cnt = {write_cnt} to file: {all_qa_results_fp}")

        return all_qa_results_fp

    def compute_acc_post(
            self,
            result_filename: str,
    ) -> None:
        result_filepath = os.path.join(self.data_dir_results, result_filename)
        assert os.path.isfile(result_filepath)
        with open(result_filepath, "r", encoding="utf-8") as fp_in:
            all_qa_results = json.load(fp_in)

        choice_set = {"A", "B", "C", "D", "E"}
        all_answers = [cur_qa_res["answer"] for cur_qa_res in all_qa_results]
        all_choices = [cur_qa_res["model_choice"] for cur_qa_res in all_qa_results]
        all_answers = [_item.strip() for _item in all_answers if isinstance(_item, str)]
        all_choices = [_item.strip() for _item in all_choices if isinstance(_item, str)]
        no_answer_cnt = len([_item for _item in all_choices if _item not in choice_set])

        try:
            assert len(all_answers) == len(all_choices)
            all_answers_np = np.array(all_answers)
            all_choices_np = np.array(all_choices)
            acc = np.mean(all_answers_np == all_choices_np).item()
            if self.verbose:
                self.logger.info(f">>> [#item = {len(all_choices)}] Accuracy: {acc:.5f}   "
                                 f"no_answer_cnt = {no_answer_cnt}")
        except Exception as e:
            if self.verbose:
                self.logger.info(e)

        # Answer token length statistics
        vlm_model = VLM(
            verbose=self.verbose, logger=self.logger, cuda_dict=self.cuda_dict,
            cache_dir=self.cache_dir, project_root_dir=self.project_root_dir,
            hf_id=self.hf_id_vlm, bsz=self.bsz,
            show_generation=self.show_generation, debug=self.debug,
            load_model=False,
        )
        all_outputs = [cur_qa_res["model_output"] for cur_qa_res in all_qa_results]
        all_outputs = [_item.strip() for _item in all_outputs if isinstance(_item, str)]
        all_outputs_len = [len(vlm_model.tokenizer_gen(_o)["input_ids"]) for _o in all_outputs]
        o_len, o_min, o_max, o_mean, o_std = (len(all_outputs_len), np.min(all_outputs_len), np.max(all_outputs_len),
                                              np.mean(all_outputs_len), np.std(all_outputs_len))
        if self.verbose:
            self.logger.info(f">>> [#item = {o_len}] Output length statistics: "
                             f"o_min = {o_min}; o_max = {o_max}; o_mean = {o_mean:.2f}; o_std = {o_std:.2f}")

        """
        --results_filename "all_qa_results-0_0_0_0_0-post.json"
        >>> [#item = 63] Accuracy: 0.49206   no_answer_cnt = 4
        >>> [#item = 63] Output length statistics: o_min = 13; o_max = 201; o_mean = 88.73; o_std = 49.40

        --results_filename "all_qa_results-1_0_0_0_0-post.json"
        >>> [#item = 63] Accuracy: 0.52381   no_answer_cnt = 3
        >>> [#item = 63] Output length statistics: o_min = 2; o_max = 201; o_mean = 90.46; o_std = 58.61

        --results_filename "all_qa_results-0_1_0_0_0-post.json"
        >>> [#item = 63] Accuracy: 0.71429   no_answer_cnt = 1
        >>> [#item = 63] Output length statistics: o_min = 2; o_max = 201; o_mean = 87.90; o_std = 40.00

        --results_filename "all_qa_results-1_1_0_0_0-post.json"
        >>> [#item = 63] Accuracy: 0.66667   no_answer_cnt = 1
        >>> [#item = 63] Output length statistics: o_min = 2; o_max = 201; o_mean = 55.76; o_std = 57.64
        """

        return None


def main(
    task: int = 0,
    verbose: bool = False,
    seed: int = 42,
    cuda: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    hf_id_vlm: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    bsz: int = 1,
    add_ds_info: bool = False,
    add_code: bool = False,
    remove_comments: bool = False,
    use_cot: bool = False,
    few_shot: int = 0,
    show_generation: bool = False,
    result_filename: str = None,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Run the experiment on the chart QA and chart captioning tasks with or without code as input.

    :param task: the task of the current run session.
    :param verbose: Verbose mode: show logs.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param hf_id_vlm: The Hugging Face model ID of VLM. Format: ORGANIZATION_NAME/MODEL_NAME
    :param bsz: The batch size.
    :param add_ds_info: Add dataset information as input or not.
    :param add_code: Add the visualization code as input or not.
    :param remove_comments: Remove the code comments or not.
    :param use_cot: Use chain-of-thought prompting or not.
    :param few_shot: The number of examples used in the few shot generation.
    :param show_generation: Whether to show outputs during generation.
    :param result_filename: The QA results file for computing accuracy.
    :param debug: Debugging / developing mode.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Code4Chart")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])
    logger.info(f">>> cuda_dict:\n{cuda_dict}")

    c4c_exp = Code4ChartExp(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        hf_id_vlm=hf_id_vlm,
        bsz=bsz,
        show_generation=show_generation,
        debug=debug,
    )

    task = int(task)
    match task:
        case 1:
            c4c_exp.run_chart_qa(
                add_ds_info=add_ds_info,
                add_code=add_code,
                remove_comments=remove_comments,
                use_cot=use_cot,
                few_shot=few_shot,
            )
        case 2:
            c4c_exp.compute_acc_post(
                result_filename=result_filename,
            )
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
