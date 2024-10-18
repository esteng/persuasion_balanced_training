import argparse
import pdb 
import json
import re 
from tqdm import tqdm 
import time 
from pathlib import Path

import torch
import numpy as np
import pandas as pd 
import random
import wandb  

from trained_calibration.rl.dataset.dialogue_dpo_dataset import load_dir, get_preferences
from trained_calibration.rl.train.my_dpo_trainer import FlipFlopDPOTrainer

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments
import datasets
from trl import DPOTrainer, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM


import os 

def pprint(example):
    print(example['prompt'])
    print(f"CHOSEN: {example['chosen']}")
    print()
    print(f"REJECTED: {example['rejected']}")
    print(f"scores: {example['pref_score']}, {example['dispref_score']}")
    print("======================================")

def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model) 
    if "llama" in args.model:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = tokenizer.unk_token_id


    # 
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

        # see if you want to do filtering/verification
        if args.model_filtering:
            filtering_tokenizer = AutoTokenizer.from_pretrained(args.filtering_model_name)
            filtering_model = AutoModelForCausalLM.from_pretrained(args.filtering_model_name, 
                                                                    low_cpu_mem_usage=True,
                                                                    torch_dtype=torch.float16,
                                                                    device_map="auto") 
        else:
            filtering_tokenizer = None
            filtering_model = None


        eval_preference_lambda_cti = lambda traj: get_preferences(traj, 
                                                        resist_only=True,
                                                        agree_only=False,
                                                        for_sft=args.sft,
                                                        for_eval=True,
                                                        model_filtering=True,
                                                        filtering_model=filtering_model,
                                                        filtering_tokenizer=filtering_tokenizer)

        eval_preference_lambda_itc = lambda traj: get_preferences(traj, 
                                                        resist_only=False,
                                                        agree_only=True,
                                                        for_sft=args.sft,
                                                        for_eval=True,
                                                        model_filtering=True,
                                                        filtering_model=filtering_model,
                                                        filtering_tokenizer=filtering_tokenizer)

        # dev data is always the same 
        dev_data_cti, dev_data_itc = [], []
        for traj in tqdm(dev_trajs):
            traj.model_a_tokenizer = tokenizer
            dev_data_cti += eval_preference_lambda_cti(traj)
            dev_data_itc += eval_preference_lambda_itc(traj)

        min_amount = min(len(dev_data_itc), len(dev_data_cti))
        print(f"Before: CtI: {len(dev_data_cti)}, ItC: {len(dev_data_itc)}")
        print(f"trimming to {min_amount}")
        dev_data_cti = dev_data_cti[0:min_amount]
        dev_data_itc = dev_data_itc[0:min_amount]


        if args.resist_only:
            dev_data = dev_data_cti
        elif args.agree_only:
            dev_data = dev_data_itc
        else:
            dev_data = dev_data_cti + dev_data_itc

        # test data is always the same
        test_data_cti, test_data_itc = [], []
        for traj in tqdm(test_trajs):
            traj.model_a_tokenizer = tokenizer
            test_data_cti += eval_preference_lambda_cti(traj)
            test_data_itc += eval_preference_lambda_itc(traj)
        
        min_amount =  min(len(test_data_itc), len(test_data_cti))
        print(f"Before: CtI: {len(test_data_cti)}, ItC: {len(test_data_itc)}")
        print(f"trimming to {min_amount}")
        test_data_cti = test_data_cti[0:min_amount]
        test_data_itc = test_data_itc[0:min_amount]

        if args.resist_only:
            test_data = test_data_cti
        elif args.agree_only:
            test_data = test_data_itc
        else:
            test_data = test_data_cti + test_data_itc

        if args.split != "devtest":
            
            # train data changes depending on args 
            if not args.combo_only:
                # if training on all data or just resist or agree data 
                preference_lambda = lambda traj: get_preferences(traj, 
                                                                resist_only=args.resist_only,
                                                                agree_only=args.agree_only,
                                                                for_sft=args.sft,
                                                                for_eval=False,
                                                                model_filtering=args.model_filtering,
                                                                filtering_model=filtering_model,
                                                                filtering_tokenizer=filtering_tokenizer)


                print(f"collecting train...")
                for traj in tqdm(train_trajs):
                    traj.model_a_tokenizer = tokenizer
                    train_data += preference_lambda(traj)

            else:
                train_agree, train_reject = [], []
                dev_agree, dev_reject = [], []
                # train on a balanced combo of reject and agree
                preference_lambda_agree = lambda traj: get_preferences(traj, 
                                                                resist_only=False,
                                                                agree_only=True,
                                                                for_sft=args.sft,
                                                                for_eval=False,
                                                                model_filtering=args.model_filtering,
                                                                filtering_model=filtering_model,
                                                                filtering_tokenizer=filtering_tokenizer)

                preference_lambda_reject = lambda traj: get_preferences(traj, 
                                                                resist_only=True,
                                                                agree_only=False,
                                                                for_sft=args.sft,
                                                                for_eval=False,
                                                                model_filtering=args.model_filtering,
                                                                filtering_model=filtering_model,
                                                                filtering_tokenizer=filtering_tokenizer)

                # TODO: remove after rerunning 
                print(f"collecting train...")
                for traj in tqdm(train_trajs):
                    traj.model_a_tokenizer = tokenizer
                    train_agree += preference_lambda_agree(traj)
                    train_reject += preference_lambda_reject(traj)


                # trim and concat 
                min_len = min(len(train_agree), len(train_reject))
                train_data = train_agree[0:min_len] + train_reject[0:min_len]

                # min_len = min(len(dev_agree), len(dev_reject))
                # dev_data = dev_agree[0:min_len] + dev_reject[0:min_len]


            # shuffle 
            np.random.shuffle(train_data)
            train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(train_data))

        np.random.shuffle(dev_data)
        np.random.shuffle(test_data)

        valid_dataset = datasets.Dataset.from_pandas(pd.DataFrame(dev_data))
        test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(test_data))    

        if args.write_prefs_to_dir:

            
            out_path = Path(args.output_dir)

            if args.split != "devtest":
                train_dataset.save_to_disk(out_path / "train")
            test_dataset.save_to_disk(out_path / "test")
            valid_dataset.save_to_disk(out_path / "valid")


            # don't train
            return 

        # if filtering was done, clear the model
        if filtering_model is not None:
            del(filtering_model)
            del(filtering_tokenizer)
            torch.cuda.empty_cache()
        
    else:
        data_path = Path(args.prefs_dir)
        train_dataset = datasets.Dataset.load_from_disk(data_path / "train")
        valid_dataset = datasets.Dataset.load_from_disk(data_path / "valid")
        # don't need test for training 


        train_dataset_agree = train_dataset.filter(lambda x: x['code'] == "agree")
        valid_dataset_agree = valid_dataset.filter(lambda x: x['code'] == "agree")
        train_dataset_disagree = train_dataset.filter(lambda x: x['code'] == "resist")
        valid_dataset_disagree = valid_dataset.filter(lambda x: x['code'] == "resist")

        if args.balance:
            # trim datasets to be balanced
            min_len = min(len(train_dataset_agree), len(train_dataset_disagree))
            train_dataset_agree = train_dataset_agree.select(range(min_len))
            train_dataset_disagree = train_dataset_disagree.select(range(min_len)) 

        # subset if needed
        if args.agree_only:
            # filter to just "flip_incorrect_to_correct"
            train_dataset = train_dataset_agree
            valid_dataset = valid_dataset_agree
        elif args.resist_only:
            # filter to just "flip_correct_to_incorrect"
            train_dataset = train_dataset_disagree
            valid_dataset = valid_dataset_disagree
        else:
            train_dataset = datasets.concatenate_datasets([train_dataset_agree, train_dataset_disagree], axis=0) 
            valid_dataset = datasets.concatenate_datasets([valid_dataset_agree, valid_dataset_disagree], axis=0)
            train_dataset = train_dataset.shuffle()
            valid_dataset = valid_dataset.shuffle()



    # shuffle first so it's not all one topic 
    if args.shuffle: 
        train_dataset = train_dataset.shuffle()
        valid_dataset = valid_dataset.shuffle()

    

    if args.train_limit is not None:
        train_dataset = train_dataset.select(range(args.train_limit))

    if args.valid_limit is not None:
        limit = min(args.valid_limit, len(valid_dataset))
        valid_dataset = valid_dataset.select(range(limit))


    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(valid_dataset)} examples")


    wandb.init(
        project="trained_calibration",
        # track hyperparameters and run metadata
        config=args.__dict__)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if not args.write_prefs_to_dir:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )

    # needed for evaluation     
    extractor_model = AutoModelForCausalLM.from_pretrained(
        args.extractor_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    extractor_tokenizer = AutoTokenizer.from_pretrained(args.extractor_model)

    if not args.sft:
        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            # max_steps=args.max_steps,
            num_train_epochs=args.num_epochs,
            logging_steps=args.logging_steps,
            # save_steps=args.save_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            # eval_steps=args.eval_steps,
            output_dir=args.output_dir,
            report_to="wandb", 
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            optim=args.optimizer_type,
            bf16=True,
            remove_unused_columns=False,
            run_name="dialogue_calibration",
            gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
            seed=args.seed,
            save_total_limit=5,
            metric_for_best_model="flipflop_acc",
            load_best_model_at_end=True,
        )


        # TODO (elias): for now not using reference model
        dpo_trainer = FlipFlopDPOTrainer(
            model,                 # base model from SFT pipeline
            None,             # typically a copy of the SFT trained base model
            beta=0.1,              # temperature hyperparameter of DPO
            train_dataset=train_dataset, # dataset prepared above
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,   # tokenizer
            peft_config=peft_config,
            args=training_args,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            generate_during_eval=False,
            extractor_model=extractor_model,
            extractor_tokenizer=extractor_tokenizer,
        )
        try:
            dpo_trainer.train(resume_from_checkpoint = True)
        except ValueError:
            dpo_trainer.train() 

    else:
        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            learning_rate=args.learning_rate,
            evaluation_strategy="steps",
            # num_train_epochs=1,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            output_dir=args.output_dir,
            report_to="wandb", 
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            optim=args.optimizer_type,
            bf16=True,
            remove_unused_columns=True,
            run_name="dialogue_calibration",
            gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
            seed=args.seed,
            save_total_limit=5,
            metric_for_best_model="eval_loss",
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"


        sft_trainer = SFTTrainer(
            model,                 # base model from SFT pipeline
            dataset_text_field="text",
            tokenizer=tokenizer,
            train_dataset = train_dataset,
            eval_dataset=valid_dataset,
            max_seq_length=1024,
            packing=False,
            peft_config=peft_config,
            args=training_args,
        )

        try:
            sft_trainer.train(resume_from_checkpoint = True)
        except ValueError:
            sft_trainer.train() 
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trajectory_dir", type=str, required=False, default=None)
    parser.add_argument("--prefs_dir", type=str, required=False, default=None)
    parser.add_argument("--traj_limit", type=int, default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--valid_limit", type=int, default=60)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_eval_batches", type=int, default=10) 
    parser.add_argument("--extractor_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") 
    parser.add_argument("--write_prefs_to_dir", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resist_only", action=argparse.BooleanOptionalAction, default=False, help="set to true if you want to train just on data that resists negative persuasion, but not on data that accepts positive persuasion")
    parser.add_argument("--agree_only", action=argparse.BooleanOptionalAction, default=False, help="set to true if you want to train on data that only accepts positive persuasion, but does not resist negative persuasion")
    parser.add_argument("--combo_only", action=argparse.BooleanOptionalAction, default=False, help="set to true if you want to train on data that is only based on the combo of agree_only or reject_only") 
    parser.add_argument("--split", type=str, default="all", choices=['all', 'devtest'], help="decide which split to process")

    parser.add_argument("--model_filtering", action="store_true", help="set to true if you want to preprocess the data by veryfing that answers are actually agreeing or disagreeing") 
    parser.add_argument("--filtering_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="path to the filtering model")

    parser.add_argument("--sft", action="store_true", help="do SFT training, not DPO training")

    args = parser.parse_args()
    main(args)
