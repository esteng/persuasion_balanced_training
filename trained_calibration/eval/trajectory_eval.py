import pdb 
from pathlib import Path 
import argparse 
from tqdm import tqdm 
import numpy as np 
import json 

from transformers import AutoModelForCausalLM, AutoTokenizer

from trained_calibration.rl.dataset.dialogue_dpo_dataset import load_dir
from trained_calibration.rl.dataset.utils import get_acc, handle_agree, handle_disagree
from trained_calibration.eval.triviaqa_evaluation import normalize_answer


from trained_calibration.rl.dataset.postprocess import postprocess_extract
from trained_calibration.eval.flipflop_eval import clean_answer

def main(args):
    if args.do_postprocess:
        extract_model = AutoModelForCausalLM.from_pretrained(args.extract_model, cache_dir="/nas-ssd2/esteng/.cache", device_map="auto")
        extract_tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
        extract_tokenizer.pad_token = extract_tokenizer.eos_token
        extract_model = extract_model.eval()
        extract_model = extract_model.half().to("cuda")

    trajs = load_dir(args.dir, remove_agree=False)

    accs = []
    model_a_acc = []
    model_b_acc = []

    for i, traj in tqdm(enumerate(trajs), total=len(trajs)): 
        try:
            if args.baseline:
                # get node before any discussion
                prev_node = traj.tree[1]
                final_node = traj.tree[2] 
            else:
                prev_node = traj.tree[-2]
                final_node = traj.tree[-1]

            if args.do_postprocess:
                prompt = traj.tree[0].generation.strip()
                if prompt[-1] == "?":
                    prompt = prompt[:-1]
                last_generation = prompt + "?\n" + final_node.generation
                last_agent_idx = 1-final_node.agent_idx

                prev_generation = prompt + "?\n" +  prev_node.generation
                prev_agent_idx = 1-prev_node.agent_idx  
                prompts = [traj.tree[0].generation, traj.tree[0].generation]
                out_responses_final, out_answers, __ = postprocess_extract(prompts, [last_generation, prev_generation], extract_model, extract_tokenizer, "trivia_qa")

                final_node.extracted_ans[last_agent_idx] = out_answers[0]
                final_node.extracted_ans[prev_agent_idx] = out_answers[1]

        except IndexError:
            # don't modify the output
            pass


        final_node = handle_disagree(traj, final_node)
        answers = final_node.extracted_ans
        answers = handle_agree(answers)
        for i, a in enumerate(answers):
            if a is None:
                answers[i] = "NONE"
        
        answers = [clean_answer(x) for x in answers]
        answers = [normalize_answer(x) for x in answers]

        gold_answers = traj.metadata['answer']['normalized_aliases']

        model_a_answer = answers[1]
        model_a_correct = get_acc(model_a_answer, gold_answers)
        model_a_acc.append(model_a_correct)
        # print(model_a_answer, gold_answers, model_a_correct)

        model_b_answer = answers[0]
        model_b_correct = get_acc(model_b_answer, gold_answers)
        # print(model_b_answer, gold_answers, model_b_correct)
        # print()
        model_b_acc.append(model_b_correct)


        corrects = [model_a_correct, model_b_correct]
        accs.append(np.mean(corrects))

    print(f"Overall Mean: {np.mean(accs)}")
    print(f"{args.model_a} Mean: {np.mean(model_a_acc)}")
    print(f"{args.model_b} Mean: {np.mean(model_b_acc)}")

    dir_path = Path(args.dir)
    if args.baseline:
        fname = dir_path/"stats_baseline.json"
    else:
        fname = dir_path/"stats.json"

    with open(fname, "w") as f1:
        to_write = {
            "overall": np.mean(accs),
            "model_a": np.mean(model_a_acc),
            "model_b": np.mean(model_b_acc)
        }
        json.dump(to_write, f1)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--model_a", type=str, default=None)
    parser.add_argument("--model_b", type=str, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--do_postprocess", action="store_true")
    parser.add_argument("--extract_model", type=str, default="/data/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b/")
    args = parser.parse_args()

    main(args)