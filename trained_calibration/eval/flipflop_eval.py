
import argparse
import pdb 
import json
import re 
from tqdm import tqdm 
import time 
from pathlib import Path
from collections import defaultdict
import datasets 

import torch
import numpy as np
import pandas as pd 
import random
import wandb  

from trained_calibration.rl.dataset.dialogue_dpo_dataset import load_dir, get_preferences
from trained_calibration.rl.dataset.utils import get_acc, handle_agree
from trained_calibration.rl.dataset.trajectory import Trajectory, Turn
from trained_calibration.eval.utils import identify_best_checkpoint

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments

def clean_text(text):
    text = re.sub("assistant\n\n", "", text)
    text = re.sub("user\n\n", "", text)
    return text.strip()

def clean_answer(answer):
    answer = answer.strip()
    answer = re.split("\(", answer)[0].strip()
    return answer

def evaluate(tokenizer, 
             model, 
             extractor_tokenizer,
             extractor_model,
             dataset, 
             batch_size,
             use_dataloader=False):

    out_data = []
    # iterate thru dataset in chunks 
    if not use_dataloader: 
        for chunk_start in tqdm(range(0, len(dataset), batch_size), total = len(dataset) // batch_size): 
            chunk = dataset[chunk_start:chunk_start+batch_size]
            batch_is_correct, batch_outputs = eval_inner_loop(tokenizer, model, extractor_tokenizer, extractor_model, chunk['prompt'], chunk['gold_answers'])
            # tokenize text 
            for i, (is_correct, prompt, generation) in enumerate(zip(batch_is_correct, chunk['prompt'], batch_outputs)):
                out_data.append({"prompt": prompt, "generation": generation, "correct": is_correct, "type": chunk['type'][i]}) 
    else:
        # in eval loop, dataset is already batched
        for batch in tqdm(dataset):
            batch_is_correct, batch_outputs = eval_inner_loop(tokenizer, model, extractor_tokenizer, extractor_model, batch['prompt'], batch['gold_answers'])
            # tokenize text 
            for i, (is_correct, prompt, generation) in enumerate(zip(batch_is_correct, batch['prompt'], batch_outputs)):
                out_data.append({"prompt": prompt, "generation": generation, "correct": is_correct, "type": batch['type'][i]}) 

    return out_data, np.mean([x['correct'] for x in out_data])


def eval_inner_loop(tokenizer, model, extractor_tokenizer, extractor_model, prompts, gold_answers):
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors='pt')
    inputs = inputs.to(model.device)

    # generate 
    outputs = model.generate(**inputs, 
                            max_new_tokens=80,
                            temperature=0.7,
                            top_p = 1.0,
                            top_k = 0.0,
                            do_sample=True,
                            min_length=15,
                            pad_token_id=tokenizer.pad_token_id) 
    # only get new tokens 
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    out_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    out_texts = [clean_text(text) for text in out_texts]

    batch_is_correct = []

    for i, generation in enumerate(out_texts): 
        # get accuracy 
        # reconstruct trajectory 
        prompt = prompts[i]
        # split prompt 
        if "llama" in model.name_or_path:
            split_prompt = re.split("<\|.*?\|>((user)|(assistant))<\|end_header_id\|>", prompt)
            split_prompt = [x.strip() for x in split_prompt if x is not None and x.strip() not in ['', "user", "assistant"]]
        else:
            # TODO implement mistral 
            eos = re.escape("[/INST]")
            sos = re.escape("[INST]")
            split_prompt = re.split(f"({sos})|({eos})", prompt)
            split_prompt = [x.strip() for x in split_prompt if x is not None and x.strip() not in ['', "<s>", "[/INST]", "[INST]"]]
            split_prompt = [re.sub(re.escape("</s>"), "", x) for x in split_prompt]
            split_prompt = [re.sub(re.escape("<s>"), "", x) for x in split_prompt]


        path = [Turn(0, 0, split_prompt[0], [None, None], None, -1, []), Turn(0, 0, split_prompt[0], [None, None], None, -1, [])]
        path.append(Turn(1, 1, generation, None, None, 0, [])) 
        dummy_traj = Trajectory([split_prompt[0]], extractor_model, extractor_tokenizer, None, None, 10, False, extract_model_idx=0)

        # pdb.set_trace()
        answers = dummy_traj.extract_ans(path, 0)
        answer = answers[1]
        answer = clean_answer(answer)
        correct_ans = gold_answers[i]

        # TODO: deal w/ agree and disagree 
        is_correct = get_acc(answer, correct_ans) 
        batch_is_correct.append(is_correct)
    return batch_is_correct, out_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--file", type=str, default=None, help="use precomputed file")
    parser.add_argument("--extract_model", type=str, required=False, default="/data/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/36d7e540e651b68dac59394d9c3381651df7fb01//")
    parser.add_argument("--trajectory_dir", type=str, required=False, default=None)
    parser.add_argument("--traj_limit", type=int, default=None)
    parser.add_argument("--prefs_dir", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--valid_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_filtering", action="store_true", help="set to true if you want to preprocess the data by veryfing that answers are actually agreeing or disagreeing") 
    parser.add_argument("--filtering_model_name", type=str, default="/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/", help="path to the filtering model")
    parser.add_argument("--split", type=str, default="valid")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path(args.model) 
    else:
        output_dir = Path(args.output_dir)

    try:
        model_path = identify_best_checkpoint(args.model)
    except:
        model_path = args.model

    if args.file is None:
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "flipflop_args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left") 
        if "llama" in model_path:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.unk_token_id


        if args.trajectory_dir is not None:
            trajs = load_dir(args.trajectory_dir) 

            if args.traj_limit is not None:
                trajs = trajs[0:args.traj_limit]


            # split into train/dev/test
            train_trajs, dev_trajs, test_trajs = (
                trajs[0:int(0.7*len(trajs))],
                trajs[int(0.7*len(trajs)):int(0.85 * len(trajs))],
                trajs[int(0.85*len(trajs)):]
            )
            train_data, dev_data, test_data = [], [], []

            dev_data_cti, dev_data_itc = [], []

            filtering_tokenizer = AutoTokenizer.from_pretrained(args.filtering_model_name, padding_side="left")
            filtering_model = AutoModelForCausalLM.from_pretrained(args.filtering_model_name, 
                                                                    low_cpu_mem_usage=True,
                                                                    torch_dtype=torch.float16,
                                                                    device_map="auto") 

            preference_lambda_cti = lambda traj: get_preferences(traj, 
                                                            resist_only=True,
                                                            agree_only=False,
                                                            for_sft=False,
                                                            for_eval=True,
                                                            model_filtering=True,
                                                            filtering_model=filtering_model,
                                                            filtering_tokenizer=filtering_tokenizer)

            preference_lambda_itc = lambda traj: get_preferences(traj, 
                                                            resist_only=False,
                                                            agree_only=True,
                                                            for_sft=False,
                                                            for_eval=True,
                                                            model_filtering=True,
                                                            filtering_model=filtering_model,
                                                            filtering_tokenizer=filtering_tokenizer)



            if args.split == "valid":
                trajs_to_use = dev_trajs
            else:
                trajs_to_use = test_trajs

            for traj in tqdm(trajs_to_use):
                traj.model_a_tokenizer = tokenizer
                dev_data_cti += preference_lambda_cti(traj)
                dev_data_itc += preference_lambda_itc(traj)

            # clear filtering model
            del(filtering_model)
            del(filtering_tokenizer)
            torch.cuda.empty_cache()


            # for traj in test_trajs:
            #     traj.model_a_tokenizer = tokenizer
            #     test_data += get_preferences(traj, for_eval=True)

            # dev_flipped_data = split_data_for_acc(dev_data)

            min_amount = min(len(dev_data_itc), len(dev_data_cti))
            print(f"Before: CtI: {len(dev_data_cti)}, ItC: {len(dev_data_itc)}")
            print(f"trimming to {min_amount}")
            dev_data_cti = dev_data_cti[0:min_amount]
            dev_data_itc = dev_data_itc[0:min_amount]

            dev_flipped_data = dev_data_cti + dev_data_itc

            test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(dev_flipped_data))

            print(f"Validating on {len(test_dataset)} examples")
            # shuffle first so it's not all one topic 
            if args.valid_limit is not None:
                limit = min(args.valid_limit, len(test_dataset))
                test_dataset = test_dataset.select(range(limit))

        else: 
            data_path = Path(args.prefs_dir)
            valid_dataset = datasets.Dataset.load_from_disk(data_path / args.split)


            # for now just do one at a time
            test_dataset = valid_dataset




        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer.padding_side = "left"

        extractor_tokenizer = AutoTokenizer.from_pretrained(args.extract_model, padding_side="left")
        extractor_model = AutoModelForCausalLM.from_pretrained(args.extract_model,
                                                                low_cpu_mem_usage=True,
                                                                torch_dtype=torch.float16,
                                                                device_map="auto"
                                                            )

        out_data, acc = evaluate(tokenizer, model, extractor_tokenizer, extractor_model, test_dataset, args.batch_size)

        print(f"OVERALL ACCURACY: {acc*100:.2f}")

        acc_on_cti = np.mean([x['correct'] for x in out_data if x['type'] == "flip_correct_to_incorrect"]) 
        acc_on_itc = np.mean([x['correct'] for x in out_data if x['type'] == "flip_incorrect_to_correct"]) 
        print(f"ACCURACY (correct to incorrect): {acc_on_cti*100:.2f}")
        print(f"ACCURACY (incorrect to correct): {acc_on_itc*100:.2f}")

        with open(output_dir / "flipflop_output.jsonl", "w") as f1:
            for line in out_data:
                f1.write(json.dumps(line) + '\n')

        with open(output_dir / 'flipflop_stats.json', 'w') as f1:
            to_write = {"acc": acc,
                        "acc_on_cti": acc_on_cti, 
                        "acc_on_itc": acc_on_itc}
            
            json.dump(to_write, f1)

    else:
        with open(args.file) as f1:
            data = [json.loads(line) for line in f1.readlines()]

        corrects = [x['correct'] for x in data]
        acc = np.mean(corrects)

        print(f"OVERALL ACCURACY: {acc*100:.2f}")

        acc_on_cti = np.mean([x['correct'] for x in data if x['type'] == "flip_correct_to_incorrect"]) 
        acc_on_itc = np.mean([x['correct'] for x in data if x['type'] == "flip_incorrect_to_correct"]) 
        print(f"ACCURACY (correct to incorrect): {acc_on_cti*100:.2f}")
        print(f"ACCURACY (incorrect to correct): {acc_on_itc*100:.2f}")