import argparse
import re 
import json
from pathlib import Path
from typing import List
import pdb 
from tqdm import tqdm

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from trained_calibration.eval.flipflop_eval import clean_text, get_acc
from trained_calibration.rl.dataset.trajectory import Trajectory, Turn
from trained_calibration.eval.utils import identify_best_checkpoint

# read predictions from file and prepare data
# def get_predictions_and_data(pred_file, data_file):
    # pass

def get_split_prompt(prompt, is_llama):
    if is_llama:
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
    return split_prompt

def get_answers(prompt, out_texts, extractor_tokenizer, extractor_model, is_llama):
    answers_to_ret = []
    for i, generation in enumerate(out_texts): 
        # get accuracy 
        # reconstruct trajectory 
        # split prompt 
        split_prompt = get_split_prompt(prompt, is_llama) 


        path = [Turn(0, 0, split_prompt[0], [None, None], None, -1, []), Turn(0, 0, split_prompt[0], [None, None], None, -1, [])]
        path.append(Turn(1, 1, generation, None, None, 0, [])) 
        dummy_traj = Trajectory([split_prompt[0]], extractor_model, extractor_tokenizer, None, None, 10, False, extract_model_idx=0)

        answers = dummy_traj.extract_ans(path, 0)
        answer = answers[1]
        answers_to_ret.append(answer)
    return answers_to_ret

def get_predictions_and_data(existing_idxs: List[int],
                             out_texts: List[str], 
                             dataset: datasets.Dataset, 
                             extract_tokenizer: AutoTokenizer, 
                             extract_model: AutoModelForCausalLM, 
                             orig_tokenizer: AutoTokenizer,
                             orig_model: AutoModelForCausalLM,
                             tokenizer: AutoTokenizer,
                             model: AutoModelForCausalLM,
                             is_llama: bool):

    final_data = []
    skipped = 0

    for i, out_text in tqdm(enumerate(out_texts), total = len(out_texts)):
        # if existing idxs are passed, only run on those where it is successful (saves time) 
        if len(existing_idxs) > 0 and i not in existing_idxs:
            continue

        datapoint = dataset[i]

        # get current answer 
        prompt = datapoint['prompt']
        split_prompt = get_split_prompt(prompt, is_llama)
        penult_gen = split_prompt[-2]
        last_gen = split_prompt[-1]
        # get the answer before the turn
        prev_answer = get_answers(prompt, [penult_gen], extract_tokenizer, extract_model, is_llama)[0]
        # get the answer from the turn
        answer = get_answers(prompt, [last_gen], extract_tokenizer, extract_model, is_llama)[0]
        # get the answer after the turn
        next_answer = get_answers(prompt, [out_text], extractor_tokenizer, extractor_model, is_llama)[0]

        # skip, extractor error
        if "NONE" in [prev_answer, answer, next_answer]:
            skipped += 1
            continue
        if "agree" in [x.lower() for x in [prev_answer, answer, next_answer]] or "disagree" in [x.lower() for x in [prev_answer, answer, next_answer]]:
            skipped += 1
            continue 
        # check for accidental same 
        if prev_answer.lower() in answer.lower() or answer.lower() in prev_answer.lower():
            skipped += 1
            continue

        # flipped if answer and subsequent answer are the same 
        if answer.strip().lower() == next_answer.strip().lower():
            flipped = True
        # not flipped if prev_answer and subsequent_answer are the same
        elif prev_answer.strip().lower() == next_answer.strip().lower():
            flipped = False
        else:
            # if neither is true, skip for now 
            skipped += 1
            continue

        answer_set = get_diversity(orig_tokenizer, orig_model, extract_tokenizer, extract_model, prompt, n_generations=20)

        prob_ratio, alternate_prob, orig_prob = get_alternate_ratio(prompt, tokenizer, model, prev_answer, answer, is_llama)


        correct_ans = datapoint['gold_answers']
        is_correct = get_acc(next_answer, correct_ans) 
        instance_type = datapoint['type']

        final_data.append({"idx": i,
                            "prev_answer": prev_answer,
                            "answer": answer,
                            "next_answer": next_answer,
                           "answer_set": answer_set,
                           "prob_ratio": prob_ratio,
                           "alternate_prob": alternate_prob,
                            "orig_prob": orig_prob,
                           "flipped": flipped,
                           "is_correct": is_correct,
                           "instance_type": instance_type})


    return final_data



# get diversity of the original model 

def get_diversity(orig_tokenizer, orig_model, extract_tokenizer, extract_model, prompt, n_generations=20): 
    # get answer distribution from original model 
    inputs = orig_tokenizer(prompt, padding=True, truncation=True, max_length=1024, return_tensors='pt')
    inputs = inputs.to(orig_model.device)

    # generate 
    outputs = orig_model.generate(**inputs, 
                            max_new_tokens=80,
                            temperature=0.7,
                            top_p = 1.0,
                            top_k = 0.0,
                            do_sample=True,
                            min_length=15,
                            num_return_sequences=n_generations,
                            pad_token_id=orig_tokenizer.pad_token_id) 
    # only get new tokens 
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    out_texts = orig_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    out_texts = [clean_text(text) for text in out_texts]

    # extract 
    is_llama = "llama" in orig_model.config._name_or_path
    answers = get_answers(prompt, out_texts, extract_tokenizer, extract_model, is_llama)
    
    return answers


# get alternate answer and listener probability of the original and alternate answer
def get_alternate_ratio(prompt: str, 
                        tokenizer: AutoTokenizer,
                        model: AutoModelForCausalLM, 
                        orig_answer: str,
                        alternate_answer: str,
                        is_llama: bool):
    split_prompt = get_split_prompt(prompt, is_llama)
    question_part = split_prompt[0]
    # format prompt 

    prompt_messages = [{"content": f"{question_part}", "role": "user"},
                       {"content": "Final answer:", "role": "assistant"}]
    prompt = tokenizer.apply_chat_template(prompt_messages, return_tensors="pt", tokenize=False)
    if is_llama:
        eot_tok = "<|eot_id|>"
        prompt = prompt[0:-len(eot_tok)]
    else:
        raise ValueError
    
    prompt_only_ids = tokenizer.encode(prompt, return_tensors="pt")
    length_to_cut = len(prompt_only_ids[0]) 

    # modify prompt 
    orig_prompt = prompt + f" {orig_answer}"
    alternate_prompt = prompt + f" {alternate_answer}"

    # tokenize prompt 
    orig_input_ids = tokenizer.encode(orig_prompt, return_tensors="pt")   
    orig_input_ids = orig_input_ids.to(model.device)
    alternate_input_ids = tokenizer.encode(alternate_prompt, return_tensors="pt")
    alternate_input_ids = alternate_input_ids.to(model.device)


    # score and get prob orig and prob alternate
    orig_answer_ids = orig_input_ids[:, length_to_cut:]
    alternate_answer_ids = alternate_input_ids[:, length_to_cut:]
    orig_output = model(orig_input_ids, labels=orig_input_ids, return_dict=True)
    alternate_output = model(alternate_input_ids, labels=alternate_input_ids, return_dict=True)

    orig_probs = torch.exp(torch.log_softmax(orig_output.logits, dim=-1))[:, length_to_cut:, :]
    # get probs at answer ids
    orig_probs = orig_probs[torch.arange(orig_probs.shape[0]), torch.arange(orig_probs.shape[1]), orig_answer_ids]
    orig_prob = orig_probs.prod(dim=-1).item() / orig_probs.shape[1]

    alternate_probs = torch.exp(torch.log_softmax(alternate_output.logits, dim=-1))[:, length_to_cut:, :]
    alternate_probs = alternate_probs[torch.arange(alternate_probs.shape[0]), torch.arange(alternate_probs.shape[1]), alternate_answer_ids]
    alternate_prob = alternate_probs.prod(dim=-1).item() / alternate_probs.shape[1]

    try:
        ratio = alternate_prob / orig_prob
    except ZeroDivisionError:
        return None, None, None
    return ratio, alternate_prob, orig_prob

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefs_dir", type=str, default="data/preprocessed_filtered/big_prefs_mistral_v0.2")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--extract_model", type=str, required=False, default="/data/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/36d7e540e651b68dac59394d9c3381651df7fb01//")
    parser.add_argument("--orig_model", type=str, required=False, default="/data/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/36d7e540e651b68dac59394d9c3381651df7fb01//")
    parser.add_argument("--model", type=str, required=False, default=None) 
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if args.model is None:
        args.model = identify_best_checkpoint(str(Path(args.jsonl_file).parent )) 

    out_path = Path(args.out_path)  
    if out_path.exists() and not args.from_scratch:
        with open(args.out_path) as f:
            existing_data = json.load(f)
            try:
                existing_idxs = [x['idx'] for x in existing_data]
            except KeyError:
                existing_idxs = []
    else:
        existing_idxs = []


    # sanity check
    if "llama" in args.model.lower() and "llama" not in args.prefs_dir:
        raise ValueError("Model is a llama model but the dataset is not a llama dataset")
    if "mistral" in args.model.lower() and "mistral" not in args.prefs_dir:
        raise ValueError("Model is a mistral model but the dataset is not a mistral dataset")
    if "llama" in args.model.lower() and "llama" not in args.jsonl_file:
        raise ValueError("Model is a llama model but the jsonl file is not a llama file")
    if "mistral" in args.model.lower() and "mistral" not in args.jsonl_file:
        raise ValueError("Model is a mistral model but the jsonl file is not a mistral file")

    data_path = Path(args.prefs_dir)
    valid_dataset = datasets.Dataset.load_from_disk(data_path / args.split)

    with open(args.jsonl_file) as f1:
        prediction_data = [json.loads(x) for x in f1.readlines()]

    out_texts = [x['generation'] for x in prediction_data]


    # for now just do one at a time
    test_dataset = valid_dataset

    extractor_tokenizer = AutoTokenizer.from_pretrained(args.extract_model, padding_side="left")
    extractor_model = AutoModelForCausalLM.from_pretrained(args.extract_model,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch.float16,
                                                            device_map="auto"
                                                        )

    if args.orig_model == args.extract_model:
        orig_tokenizer = extractor_tokenizer
        orig_model = extractor_model
    else:
        orig_tokenizer = AutoTokenizer.from_pretrained(args.orig_model, padding_side="left")
        orig_model = AutoModelForCausalLM.from_pretrained(args.orig_model,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch.float16,
                                                            device_map="auto"
                                                        )
    if args.model == args.extract_model:
        tokenizer = extractor_tokenizer
        model = extractor_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    device_map="auto"
                                                )

    # set eval mode

    extractor_model.eval()
    orig_model.eval()
    model.eval() 

    # set tokenizer pad tokens
    extractor_tokenizer.pad_token = extractor_tokenizer.eos_token
    orig_tokenizer.pad_token = orig_tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token


    # get predictions and data
    final_data = get_predictions_and_data(
                             existing_idxs,
                             out_texts, 
                             test_dataset, 
                             extractor_tokenizer, 
                             extractor_model, 
                             orig_tokenizer, 
                             orig_model, 
                             tokenizer, 
                             model, 
                             "llama" in args.model.lower())
    
    # save data
    with open(args.out_path, "w") as f:
        json.dump(final_data, f)