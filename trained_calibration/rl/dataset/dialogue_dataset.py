import pdb 
import json 
import argparse
from pathlib import Path
from tqdm import tqdm 

import random
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from trained_calibration.rl.dataset.prompts import (resistant_prompt, 
                                                    acceptant_prompt, 
                                                    logical_prompt, 
                                                    emotional_prompt, 
                                                    credibility_prompt, 
                                                    standard_prompt) 
from trained_calibration.eval.triviaqa_evaluation import normalize_answer as tqa_normalize_answer
from trained_calibration.eval.utils import identify_best_checkpoint

from trained_calibration.rl.dataset.trajectory import TrajectoryFactory




def generate_trajectories(dataset_name: str,  
                          trajectory_factory: TrajectoryFactory,
                          out_path: str,
                          limit: int = 1000,
                          num_per_example: int = 5,
                          shard_size: int = None,
                          shard_idx: int = None,
                          split: str = "train"):

    if dataset_name == "trivia_qa":
        dataset = load_dataset("mandarjoshi/trivia_qa", 
                           "rc.web.nocontext") 


    elif dataset_name == "truthful_qa": 
        dataset = load_dataset("truthful_qa", "generation") 

    # shuffle dataset 
    dataset = dataset.shuffle(seed=42)
    # sample limit
    train_data = dataset[split]
    train_data = train_data.select(range(limit))
    # shard up the data
    if shard_size is not None:
        num_shards = limit // shard_size
        train_data = train_data.shard(num_shards=num_shards, index=shard_idx)

    already_done_data = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        already_done_data = []
        with open(out_path) as f1:
            for line in f1:
                try:
                    already_done_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def safe_load(x):
        try:
            return json.loads(x)
        except:
            return x
    done = [safe_load(x)['metadata']['question_id'] for x in already_done_data] 

    

    with open(out_path, 'w', 1) as f:
        for line in already_done_data:
            f.write(json.dumps(line) + "\n")

                
        for example in tqdm(train_data):
            brokens = []


            for i in range(num_per_example):

                # don't overwrite 
                full_id = f"{example['question_id']}_{i}"
                if full_id in done:
                    continue


                if split == "train": 
                    rp = resistant_prompt.format(example['question'])
                    ap = acceptant_prompt.format(example['question'])
                    lp = logical_prompt.format(example['question'])
                    ep = emotional_prompt.format(example['question'])
                    cp = credibility_prompt.format(example['question'])


                                # choice =  np.random.choice([0,1,2]) 
                    # prompt = [normal_prompt, disagree_prompt, question_prompt][choice]
                    prompts_agent_0 = [rp, ap]
                    prompts_agent_1 = [lp, ep, cp]
                    prompts = [prompts_agent_0, prompts_agent_1]
                else:
                    # at test time, do not give specific instructions 
                    prompt_a1 = standard_prompt.format(example['question'])
                    prompt_a2 = standard_prompt.format(example['question'])
                    prompts = [[prompt_a1], [prompt_a2]]

                # prompts = [certain_prompt, disagree_prompt, question_prompt]

                example['question_id'] = full_id
                metadata = example
                trajectory = trajectory_factory.build(prompts, metadata) 
                was_broken = trajectory.fill()
                brokens.append(was_broken)
                json_trajectory = trajectory.to_json()  
                json_trajectory = json.dumps(json_trajectory)
                f.write(json_trajectory + "\n")

    return dataset

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="dialogue_data_tree")
    parser.add_argument("--num_per_example", type=int, default=1)
    parser.add_argument("--both_do_first", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_a", type=str, default=None)
    parser.add_argument("--model_b", type=str, default=None)
    parser.add_argument("--extractor_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if args.model_a is None:
        model_a_path = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    else:
        model_a_path = args.model_a 

    try:
        tokenizer_a = AutoTokenizer.from_pretrained(model_a_path) 
    except:
        # identify best checkpoint when loading a locally-trained model 
        model_a_path = identify_best_checkpoint(model_a_path)
        tokenizer_a = AutoTokenizer.from_pretrained(model_a_path) 


    # added auto 
    print(f"MODEL_A PATH: {model_a_path}")

    model_a = AutoModelForCausalLM.from_pretrained(model_a_path, device_map="auto") 

    if "llama" in model_a_path.lower():
        tokenizer_a.pad_token = tokenizer_a.eos_token
    else:
        tokenizer_a.pad_token = tokenizer_a.unk_token

    model_a = model_a.half() 

    # if args.model_b is None:
    if args.model_b is None:
        model_b_path =  "mistralai/Mistral-7B-Instruct-v0.2"
    else:
        model_b_path = args.model_b

    print(f"MODEL_B PATH: {args.model_b}")
    try:
        tokenizer_b = AutoTokenizer.from_pretrained(model_b_path) 
    except:
        model_b_path = identify_best_checkpoint(model_b_path)
        tokenizer_b = AutoTokenizer.from_pretrained(model_b_path) 


    model_b = AutoModelForCausalLM.from_pretrained(model_b_path, device_map="auto")


    if "llama" in model_b_path:
        tokenizer_b.pad_token = tokenizer_b.eos_token
    else:
        tokenizer_b.pad_token = tokenizer_b.unk_token

    model_b = model_b.half()

    if args.extractor_model is not None:
        extractor_model = AutoModelForCausalLM.from_pretrained(args.extractor_model, device_map="auto") 
        extractor_tokenizer = AutoTokenizer.from_pretrained(args.extractor_model) 
        extractor_model = extractor_model.half()
        # extractor_model = extractor_model.to("cuda:0")
    else:
        extractor_tokenizer, extractor_model = None, None

    factory = TrajectoryFactory(model_a, 
                                tokenizer_a, 
                                model_b, 
                                tokenizer_b, 
                                args.max_turns, 
                                args.both_do_first, 
                                True, 
                                extractor_model=extractor_model, 
                                extractor_tokenizer=extractor_tokenizer,
                                deterministic=args.deterministic)

    out_path = Path(args.out_dir) / f"tqa_{args.limit}_shard_{args.shard_idx}.jsonl"
    generate_trajectories("trivia_qa", 
                          trajectory_factory=factory, 
                          out_path=out_path,
                          limit=args.limit,
                          num_per_example=args.num_per_example,
                          shard_size=args.shard_size,
                          shard_idx=args.shard_idx,
                          split=args.split)
