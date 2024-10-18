
from collections import defaultdict
import re 
import pdb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trained_calibration.rl.dataset.utils import handle_agree, handle_disagree, check_acc, draw_sumtree
from trained_calibration.rl.dataset.trajectory import Trajectory
from trained_calibration.rl.dataset.utils import get_acc

class PreferenceFactory:
    def __init__(self, 
                 traj: Trajectory,
                 filtering_tokenizer: AutoTokenizer = None,
                 filtering_model: AutoModelForCausalLM = None):

        self.traj = traj
        self.filtering_tokenizer = filtering_tokenizer
        self.filtering_model = filtering_model
        if self.filtering_model is not None:
            # for now, only handle llama
            assert("Llama" in self.filtering_model.name_or_path)
        if self.filtering_tokenizer is not None:
            self.filtering_tokenizer.pad_token = self.filtering_tokenizer.eos_token
            self.filtering_tokenizer.padding_side = "left"

    def get_sumtree(self):
        # compute the sum of the leaf nodes, propagating up, and prefer higher to lower sums
        sums_by_idx = defaultdict(int) 
        done_leaves = []
        
        def sum_helper(curr_node_idx):
            curr_node = self.traj.get_node_by_idx(curr_node_idx)
            # don't do leaves twice 
            if len(curr_node.children) == 0 and curr_node_idx in done_leaves:
                return 
            done_leaves.append(curr_node_idx)
            # get accuracy

            curr_node = handle_disagree(self.traj, curr_node)
            child_accs = check_acc(self.traj, curr_node)
            if child_accs is None:
                return
            sums_by_idx[curr_node_idx] += sum(child_accs)
            parent_node_idx = curr_node.parent
            if parent_node_idx == curr_node_idx or curr_node_idx == -1:
                return
            # aggregate upwards
            sums_by_idx[parent_node_idx] += sums_by_idx[curr_node_idx]

        # get all leaf nodes
        # topological sort 
        topo_sort = []
        def topo_helper(curr_node_idx):
            topo_sort.append(curr_node_idx)
            curr_node = self.traj.get_node_by_idx(curr_node_idx)
            for child in curr_node.children:
                topo_helper(child)
        topo_helper(-1)

        topo_sort.reverse()

        for node in topo_sort:
            sum_helper(node)

        return sums_by_idx


    def get_siblings(self, node_idx):
        if node_idx == -1:
            return []
        node = self.traj.get_node_by_idx(node_idx)
        parent = self.traj.get_node_by_idx(node.parent)
        sibling_idxs = [x for x in parent.children if x != node_idx]
        return sibling_idxs 

    def get_prompt(self, node):
        parent_path = self.traj.get_parent_path(node.node_idx)[0:-1]
        question = re.split("Question:", parent_path[0].generation)[1].strip()
        question = f"""You will be asked trivia questions. Please respond to the best of your ability. Your response should reflect how confident you are in your answer, and why you believe your answer is right. 
Question: {question}"""
        prompt = self.traj.model_a_tokenizer.batch_decode(self.traj.get_input_prompt(question, True, parent_path))[0]
        return prompt

    def format_example(self,
                       prefered_node_idx, 
                       disprefered_node_idx): 
        """format an example for DPO training """

        pref_node, dispref_node = (self.traj.get_node_by_idx(prefered_node_idx), 
                                    self.traj.get_node_by_idx(disprefered_node_idx))
        # sanity check, so that parent path is the same
        assert(pref_node.parent == dispref_node.parent)
        prompt = self.get_prompt(pref_node)
        try:
            question = re.search("Question: (.*)", prompt).group(1)
        except AttributeError:
            return None

        if "mistral" in self.traj.model_a_tokenizer.name_or_path.lower():
            chosen = f"{self.traj.model_a_tokenizer.bos_token}{pref_node.generation}{self.traj.model_a_tokenizer.eos_token}"
            rejected = f"{self.traj.model_a_tokenizer.bos_token}{dispref_node.generation}{self.traj.model_a_tokenizer.eos_token}"
        else:
            pref_messages = [{"content": pref_node.generation, "role": "assistant"}]
            dispref_messages = [{"content": dispref_node.generation, "role": "assistant"}]
            chosen = self.traj.model_a_tokenizer.apply_chat_template(pref_messages, tokenize=False)
            rejected = self.traj.model_a_tokenizer.apply_chat_template(dispref_messages, tokenize=False)


        return {"question": question,
                "prompt": prompt, 
                "chosen": chosen, 
                "rejected": rejected} 

    def get_sft_prompt(self, node):
        """format an example for SFT training """
        parent_path = self.traj.get_parent_path(node.node_idx)
        question = re.split("Question:", parent_path[0].generation)[1].strip()
        question = f"""You will be asked trivia questions. Please respond to the best of your ability. Your response should reflect how confident you are in your answer, and why you believe your answer is right. 
Question: {question}"""
        prompt = self.traj.get_input_prompt(question, True, parent_path, tokenize=False)
        return question, prompt

    def format_for_sft(self, preferred_node_idx):
        pref_node = self.traj.get_node_by_idx(preferred_node_idx)
        question, text = self.get_sft_prompt(pref_node)

        return {"question": question,
                "text": text} 
    


    def filter_by_agree_or_resist(self, pref_node, dispref_node):
        # if resist, only include examples where the preferred example resists negative persuasion
        # and the dispreferred one accepts negative persuasion 

        # if accept, only include examples where the preferred example accepts positive persuasion
        # and the dispreferred one rejects positive persuasion 

        # get pref node parent (same as dispref_node parent)
        parent_idx = pref_node.parent
        parent_node = self.traj.get_node_by_idx(parent_idx)

        # skip if the agents haven't both answered
        if None in parent_node.extracted_ans:
            return None


        # parent is the persuasion turn
        parent_node.extracted_ans = handle_agree(parent_node.extracted_ans)
        pref_node.extracted_ans = handle_agree(pref_node.extracted_ans)
        dispref_node.extracted_ans = handle_agree(dispref_node.extracted_ans)

        parent_corrects = check_acc(self.traj, parent_node)
        pref_corrects = check_acc(self.traj, pref_node)
        dispref_corrects = check_acc(self.traj, dispref_node)


        # last generation in the prompt is the "flipping" generation 
        # children are preferred/dispreferred nodes that are either correct or incorrect 
        # two cases:
        # 1. if preferred node stayed same as parent and dispreferred node became less correct, then flipping from correct to incorrect
        # 2. if preferred node was more correct than parent and dispreferred is less, then flipping incorrect to correct 

        if parent_corrects is None or pref_corrects is None or dispref_corrects is None:
            return None

        parent_sum = sum(parent_corrects)
        pref_sum = sum(pref_corrects)
        dispref_sum = sum(dispref_corrects)

        grandparent_idx = parent_node.parent
        grandparent_node = self.traj.get_node_by_idx(grandparent_idx)
        curr_agent_idx = pref_node.agent_idx
        grandparent_answer = grandparent_node.extracted_ans[curr_agent_idx]

        grandparent_corrects = check_acc(self.traj, grandparent_node)
        if grandparent_corrects is None:
            return None

        grandparent_correct = grandparent_corrects[grandparent_node.agent_idx]

        pref_correct = pref_corrects[pref_node.agent_idx]
        dispref_correct = dispref_corrects[dispref_node.agent_idx]

        parent_correct = parent_corrects[parent_node.agent_idx]

        if grandparent_answer is None:
            # can't have been convinced or not by the parent node 
            return None

        if parent_sum == pref_sum: # automatically means dispreferred is less  
            # this is a disagree only example
            # grandparent was right, parent tried to flip, preferred refused the flip 
            if (grandparent_correct and 
                # not parent_correct and 
                pref_correct and 
                not dispref_correct):
                return "resist"

        if parent_sum < pref_sum: # this is a potential agree only example
            # grandparent was wrong, parent tried to flip, preferred accepted the flip 
            # parent not needed for now 
            if (not grandparent_correct and 
                # parent_correct and 
                pref_correct and 
                not dispref_correct): 
                return "agree"

        return None
    
    def do_model_based_agreement(self): 
        """go through all nodes and reannotate answers to see if they disagree or not"""

        prompts = []
        final_responses_by_node_idx = {}

        question = re.split("Question:", self.traj.tree[0].generation)[1].strip()
        idxs = []
        for i, node in enumerate(self.traj.tree):
            answers = node.extracted_ans
            if None in answers:
                final_responses_by_node_idx[i] = False
                continue 

            prompt = f"""Given are two answers to a question. Please determine if these answers differ substantially.
Give your answer in the following format: Final Response: <response>
Do not include any other text.
If the answers are substantially different, respond Final Response: disagree
If the answers are synonymous or the same, respond Final Response: agree

Here are the question and its answers:
Question: {question}
Answers: {answers[0]}, {answers[1]}"""
            prompt_messages = [{"content": prompt, "role": "user"}]
            formatted_prompt = self.filtering_tokenizer.apply_chat_template(prompt_messages, return_tensors="pt", tokenize=False)
            formatted_prompt += "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|> Final Response:"
            prompts.append(formatted_prompt)
            idxs.append(i)

        # encode and score a batch
        try:
            input_batch = self.filtering_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        except IndexError:
            # batch is empty, don't do anything with it 
            return 

        input_batch = {k:v.to(self.filtering_model.device) for k,v in input_batch.items()}

        yes_idx = self.filtering_tokenizer.encode("agree", add_special_tokens=False)[0]
        no_idx = self.filtering_tokenizer.encode("disagree", add_special_tokens=False)[0]


        output_logits = self.filtering_model(**input_batch).logits
        output_logit_slice = output_logits[:, -1, :]

        output_probs = torch.exp(torch.log_softmax(output_logit_slice, dim=-1))
        p_yes = output_probs[:, yes_idx]
        p_no = output_probs[:, no_idx]

        prob_yes = p_yes / (p_yes + p_no)
        prob_yes = [prob_yes[i] for i in range(prob_yes.shape[0])]

        del(input_batch)
        del(output_logits)
        del(output_probs)

        # agree if prob agree is greater than prob disagree    
        is_agree = [x > 0.5 for x in prob_yes]
        for idx, agg in zip(idxs, is_agree):
            final_responses_by_node_idx[idx] = agg

        # reannotate 
        for i, node in enumerate(self.traj.tree): 
            agreement = final_responses_by_node_idx[i]
            if agreement:
                # check if any answers are correct 
                accs = check_acc(self.traj, node)
                # if one answer is true and they agree, flip the incorrect answer
                if True in accs:
                    true_idx = accs.index(True) 
                    true_answer = node.extracted_ans[true_idx]
                    # make them agree w/ the true answer
                    node.extracted_ans[1-true_idx] = true_answer
                # if they're both wrong, doesn't matter 
                else:
                    node.extracted_ans[1] = node.extracted_ans[0]

                self.traj.tree[i] = node
            else:
                # leave it alone 
                continue

    def confirm_code(self, pref_node, dispref_node, code):
        # use model to confirm that the code is actually accurate
        # code is either resist, agree, or None
        if code not in ['resist', 'agree']: 
            return False
        
        # check for code 
        # should classify the response as either being an agreeing response or a disagreeing response 
        prompts = []
        for node in [pref_node, dispref_node]:
            agree_code_prompt = f"""You will see a response to a discussion. Classify the response as either being a disagreeing response or an agreeing response. Output ONLY the word "agree" or "disagree", and nothing else. 
Examples:
Response: I'm glad this disgreement doesn't come between our friendship. I still believe that India was a French colony.
Is this an "agree" response or a "disagree" response?
Final Response: disagree

Response: Looks like we are on the same page! The first man to walk on the moon was indeed Neil Armstrong.
Is this an "agree" response or a "disagree" response?
Final Response: agree
    
Response: {node.generation.strip()}
Is this an "agree" response or a "disagree" response?"""

            prompt_messages = [{"content": agree_code_prompt, "role": "user"}]
            formatted_prompt = self.filtering_tokenizer.apply_chat_template(prompt_messages, return_tensors="pt", tokenize=False)
            formatted_prompt += "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|> Final Response:"
            prompts.append(formatted_prompt)


        # encode and score a batch
        try:
            input_batch = self.filtering_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        except IndexError:
            # batch is empty, don't do anything with it 
            return 

        input_batch = {k:v.to(self.filtering_model.device) for k,v in input_batch.items()}

        agree_idx = self.filtering_tokenizer.encode("agree", add_special_tokens=False)[0]
        disagree_idx = self.filtering_tokenizer.encode("disagree", add_special_tokens=False)[0]


        output_logits = self.filtering_model(**input_batch).logits
        output_logit_slice = output_logits[:, -1, :]

        output_probs = torch.exp(torch.log_softmax(output_logit_slice, dim=-1))
        p_yes = output_probs[:, agree_idx]
        p_no = output_probs[:, disagree_idx]

        prob_yes = p_yes / (p_yes + p_no)
        prob_yes = [prob_yes[i] for i in range(prob_yes.shape[0])]

        pref_is_agree = prob_yes[0] > 0.5
        dispref_is_agree = prob_yes[1] > 0.5
        pref_is_agree = pref_is_agree.item()
        dispref_is_agree = dispref_is_agree.item()

        print(f"Pref node: {pref_node.generation}")
        print(f"pref is agree: {pref_is_agree}")
        print(f"Disref node: {dispref_node.generation}")
        print(f"dispref is agree: {dispref_is_agree}")
        print()

        # if the code originally was "agree", then pref needs to be agree and dispref needs to be disagree
        if code == "agree":
            return pref_is_agree and not dispref_is_agree

        # if the code originally was "disagree", then pref needs to be DISAGREE and dispref needs to be agree 
        if code == "resist": 
            return not pref_is_agree and dispref_is_agree


    def get_preferences(self,
                        resist_only=False, 
                        agree_only = False, 
                        for_sft=False, 
                        for_eval=False, 
                        model_filtering=False):
        """Recurse through the tree and extract preferences"""

        if model_filtering:
            self.do_model_based_agreement()


        sums_by_idx = self.get_sumtree()
        # draw_sumtree(traj, sums_by_idx)

        # go through trajectory and compare siblings to get preference data 
        # the higher we are in the tree, the bigger the difference needs to be 
        # e.g. at level 1, a difference of 18 to 20 isn't meaningful, but 10-20 is 
        # vs lower level, a difference of 4 to 2 is probably meaningful 
        # for now, let's say one option needs to be 25% better than the other to be included
        all_data = []
        done_hashes = []

        def pref_helper(curr_node_idx):  
            siblings = self.get_siblings(curr_node_idx)
            curr_node = self.traj.get_node_by_idx(curr_node_idx)
            curr_children_number = len(curr_node.children)

            if len(siblings) > 0:
                curr_score = sums_by_idx[curr_node_idx]
                sibling_scores = [(s, sums_by_idx[s]) for s in siblings]
                # print(f"score: {curr_score} siblings: {sibling_scores}")
                for sibling, score in sibling_scores:
                    sibling_node = self.traj.get_node_by_idx(sibling)
                    sibling_children_number = len(sibling_node.children)
                    if curr_children_number != sibling_children_number:
                        # skip this comparison: it's not fair to compare childless nodes since they can't aggregate upwards
                        continue 
                    
                    # equally good 
                    try:
                        if sum(check_acc(self.traj, curr_node)) == 2 and sum(check_acc(self.traj, sibling_node)) == 2:
                            continue
                    except TypeError:
                        continue

                    # see if 25% difference 
                    bigger_score = max(curr_score, score)
                    smaller_score = min(curr_score, score)
                    if not (bigger_score == smaller_score or bigger_score == 0):
                        # print(bigger_score, smaller_score)
                        if abs(bigger_score - smaller_score)/bigger_score > 0.50: 

                            (dispreferred_idx, _), (preferred_idx, _) = sorted([(curr_node_idx, curr_score), (sibling, score)], key=lambda x: x[1])
                            nodes = [curr_node, sibling_node]
                            pref_node = [n for n in nodes if n.node_idx == preferred_idx][0]
                            dispref_node = [n for n in nodes if n.node_idx == dispreferred_idx][0]

                            code = self.filter_by_agree_or_resist(pref_node, dispref_node)

                            if model_filtering:
                                confirmation_result = self.confirm_code(pref_node, dispref_node, code)
                                if not confirmation_result:
                                    continue

                            if resist_only or agree_only:
                                if resist_only and code != "resist":
                                    # only allow if it is a resisting example
                                    continue
                                elif resist_only and code == "resist":
                                    print(pref_node.generation)

                                if agree_only and code != "agree": 
                                    continue
                                elif agree_only and code == "agree":
                                    print(pref_node.generation)

                            if for_sft:
                                example = self.format_for_sft(preferred_idx)
                            else:
                                example = self.format_example(preferred_idx, dispreferred_idx)

                            if example is None:
                                continue 

                            example['code'] = code

                            keys = sorted(example.keys())
                            hash_val = hash("-".join([example[k] for k in keys]))
                            if hash_val in done_hashes:
                                continue
                            done_hashes.append(hash_val)
                            if not for_sft:
                                example['pref_score'] = bigger_score
                                example['dispref_score'] = smaller_score
                            if for_eval:

                                example['parent_answers'] = self.traj.get_node_by_idx(pref_node.parent).extracted_ans
                                example['pref_answers'] = pref_node.extracted_ans
                                example['dispref_answers'] = dispref_node.extracted_ans

                                example['gold_answers'] = self.traj.metadata['answer']['normalized_aliases']

                                if resist_only:
                                    example['type'] = "flip_correct_to_incorrect" 
                                if agree_only: 
                                    example['type'] = "flip_incorrect_to_correct" 

                            all_data.append(example)

            for child in curr_node.children: 
                pref_helper(child)
        pref_helper(-1)

        if not for_sft:
            # sort by difference 
            examples_by_question = defaultdict(list)
            for example in all_data:
                examples_by_question[example['question']].append(example)

            def get_diff_score(pref, dispref):
                perc_increase = abs(pref - dispref)/pref
                return perc_increase
            sorted_data = []
            for quest, examples in examples_by_question.items():
                examples = sorted(examples, key = lambda x: get_diff_score(x['pref_score'], x['dispref_score']))
                sorted_data += examples
        else:
            sorted_data = all_data
        return sorted_data