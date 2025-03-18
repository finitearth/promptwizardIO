### This code tries to implement the demos/gsm8k.ipynb notebook in a script format.

from argparse import ArgumentParser
import os

# has to come before imports, as we can only specify model via env variables
args = ArgumentParser()
args.add_argument("--dataset", type=str, default="openai/gsm8k")
args.add_argument("--revision", type=str, default="main")
args.add_argument("--model", type=str, default="vllm-ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G128_W4A16_MSE")
args = args.parse_args()

os.environ["MODEL"] = args.model
# import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from promptwizard.glue.common.utils.file import save_jsonlist
from typing import Any
from tqdm import tqdm
from re import compile, findall
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv(override = True)


class GSM8k(DatasetSpecificProcessing):
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            # Your functions for metrics and prompt building
            ans_re = compile(r"#### (\-?[0-9\.\,]+)")
            self.INVALID_ANS = "[invalid]"

            match = ans_re.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return self.INVALID_ANS

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
              DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
              DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
              DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        
        if not answer:
            return self.INVALID_ANS

        model_pred = answer.lower()
        preds = model_pred.split(self.ANSWER_START.lower())
        answer_flag = True if len(preds) > 1 else False

        pred = preds[-1].replace(",", "")
        pred = [s for s in findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        return pred
    

if __name__ == "__main__":
    gsm8k_processor = GSM8k()

    if not os.path.exists("data"):
        os.mkdir("data")
    
    dataset = load_dataset(args.dataset, "main")
    num_samples = 0
    for dataset_type in ['train','test']:
        data_list = []
        for data in dataset[dataset_type]:
            data_list.append({"question": data['question'], "answer": data['answer']})
            if num_samples == 100 and dataset_type == 'train': # We sample only 100 train examples and use 25 out them for training randomly
                break
            num_samples += 1
        gsm8k_processor.dataset_to_jsonl("data/"+ dataset_type+'.jsonl', dataset=data_list)

    train_file_name = os.path.join("data", "train.jsonl")
    test_file_name = os.path.join("data", "test.jsonl")
    path_to_config = "configs"
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   train_file_name,
                   gsm8k_processor)
    
    best_prompt, expert_profile = gp.get_best_prompt(use_examples=True,run_without_train_examples=False,generate_synthetic_examples=False)

    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")
