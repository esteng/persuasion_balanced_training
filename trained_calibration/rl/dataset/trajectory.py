import pdb
import re 
import json
import torch
from typing import List, Dict

import numpy as np 
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

class Turn:
    def __init__(self,
                 node_idx,
                 agent_idx,
                 generation, 
                 extracted_ans,
                 extracted_conf, 
                 parent = None,
                 children = []):
        self.node_idx = node_idx
        self.agent_idx = agent_idx
        self.generation = generation
        self.extracted_ans = extracted_ans
        self.extracted_conf = extracted_conf
        self.parent = parent 
        self.children = children


    def to_json(self):
        return self.__dict__

    @classmethod
    def from_json(cls, jsonline):
        return cls(**jsonline)


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


class Trajectory:
    def __init__(self,
                 prompts: List[List[str]],
                 model_a: AutoModelForCausalLM,
                 model_a_tokenizer: AutoTokenizer,
                 model_b: AutoModelForCausalLM,
                 model_b_tokenizer: AutoTokenizer,
                 max_turns: int,
                 both_do_first: bool,
                 extract_model_idx: int = 0,
                 metadata: Dict = None,
                 break_on_agree: bool = True,
                 extractor_model: AutoModelForCausalLM = None,
                 extractor_tokenizer: AutoTokenizer = None,
                 deterministic: bool = False):

        self.prompts = prompts
        self.model_a = model_a
        self.model_a_tokenizer = model_a_tokenizer
        self.model_b = model_b
        self.model_b_tokenizer = model_b_tokenizer
        self.max_turns = max_turns
        self.both_do_first = both_do_first
        self.extract_model_idx = extract_model_idx

        self.extractor_model = extractor_model
        self.extractor_tokenizer = extractor_tokenizer

        self.metadata = metadata
        self.deterministic = deterministic

        # pick on prompt to start with 
        prompt = prompts[0][0]
        # prompt = prompts[np.random.choice(len(prompts))]
        self.tree = [Turn(-1, -1, prompt, [None, None], -1.0, parent=-1, children=[])]
        self.break_on_agree = break_on_agree

    def alternate_messages(self, messages):
        if len(messages) == 0:
            return messages
        is_even = len(messages) % 2 == 0
        if is_even:  
            even_role = "assistant"
            odd_role = "user"
            # even_role = "user"
            # odd_role = "assistant"
        else:
            even_role = "user"
            odd_role = "assistant"
        for i in range(len(messages)):
            if i % 2 == 0:
                messages[i]["role"] = even_role
            else:
                messages[i]["role"] = odd_role
            #Mistral last message is always user
        if is_even:
            # add so that the order alternates correctly
            messages = [{"role": "user", "content": ""}] + messages
        # assert(messages[-1]['role'] == "user") 
        return messages


    def get_input_prompt(self, prompt: str, for_a: bool, path: List[Turn], tokenize: bool = True):
        # prompt is always user 
        first_content = prompt
        messages = [{"role": "user", "content": first_content}]

        if for_a: 
            tokenizer = self.model_a_tokenizer
        else:
            tokenizer = self.model_b_tokenizer
        for i, turn in enumerate(path):
            # skip prompt 
            if i == 0:
                tokenizer = self.model_a_tokenizer
                continue
            if for_a: 
                tokenizer = self.model_a_tokenizer
                if i % 2 == 0:
                    role = "user"
                else:
                    role = "assistant"
            else:
                tokenizer = self.model_b_tokenizer
                if i % 2 != 0:
                    role = "user"
                else:
                    role = "assistant"
            messages.append({"role": role, "content": turn.generation.strip()})
        if self.both_do_first:
            # if they are both supposed to do the first example, then delete messages[1] for first case of for_b
            # length is 2 because first is prompt and second is a
            if not for_a and len(messages) == 2:
                messages = messages[0:1]

        
        messages = self.alternate_messages(messages)
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=tokenize)
        return encodeds


    def get_extract_input_short(self, tokenizer: AutoTokenizer, path: List[Turn], tokenize: bool): 
        question = re.findall("Question: (.*)", path[0].generation)[-1].strip()
        response = path[-1].generation.strip()
        try:
            if response.strip()[-1] not in ['.?!']:
                # add ellipses so model doesn't keep generating response
                response = f"{response}..."
        except IndexError:
            pass

        prompt = f"""Please look at the following answer to a question and decide what the final answer should be. You'll first see the question and then see a single response from a dialogue about the question. 
Extract the final answers from this dialogue. Do NOT assess whether the answers are correct. 
Try to extract an answer of 1-2 words. If no answer can be extracted, give one of the following special tokens:
- Agree: use this token if the answer simply agrees with the previous answer without giving a new one.
    Example: Response: That's right! I'm glad we could come to an agreement on the question of who won the 1996 election!
Final Answer: Agree
- Disagree: use this token if the answer simply disagrees with the previous answer without giving a new one.
    Example: Response: I think you're wrong. The winner of the 1996 election was not Al Gore.
Final Answer: Disagree
- NONE: use this if the answer neither disagrees or agrees but does not state any answer. 
    Example: I don't know enough about the topic to give an answer, sorry. 
Final Answer: NONE

Examples: 
Question: Who wrote Paradise Lost?
Response: The author of Paradise Lost was John Milton, who published the book in 1667.
Final answer: John Milton

Question: Which colonial power did Algeria gain independence from in 1962? 
Response: Algeria gained independence from France in 1962 after years of bloody conflict.
Final answer: France

Question: How many presidents did the United States have in the 20th century?
Response: My interlocutor is clearly mistaken and should check their facts.
Final answer: Disagree

Question: Which movie star was known as the "King of Hollywood"?
Response: I'm glad we're both on the same page!
Final answer: Agree

Question: How many planets are in our solar system?
Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
Final answer: NONE

Only use these if NO answer can be extracted. If you can instead extract any answer, just report the answer and nothing else. You should never combine "Agree/Disagree/NONE" with any answer.
Give your final output as:
Final Answer: <final answer (1-2 words ONLY)>

Question: {question}
Response: {response}"""
        prompt_messages = [{'content': prompt, 
                            'role': 'user'}]

        # encoded = tokenizer.encode(prompt, return_tensors='pt')
        encoded = tokenizer.apply_chat_template(prompt_messages, return_tensors='pt', tokenize=tokenize) 
        return encoded
    
    def get_extract_input(self, path: List[Turn]):
        if self.extract_model_idx == 0:
            tokenizer = self.model_a_tokenizer
        else:
            tokenizer = self.model_b_tokenizer

        input_prompt = """Please look at the following coversation about a question and decide what the final answer should be. You'll first see the question and then see a dialogue about the question. 
Extract the final answers from this dialogue in json format. Do not assess whether the answers are correct or not. 
First, write a comment with your thinking, then give the final output in the following format: `{"answer_1": <answer from agent 1>, "answer_2": <answer from agent 2>}` for each agent and nothing else. If no answer can be extracted, the answer variable should be filled with "NONE".
Example:

Question: Who wrote Paradise Lost?

Agent 1: I believe the author of Paradise Lost was John Milton, who published the book in 1667.

Agent 2: I agree, I also think it's John Milton
# Thought: Agent 1 said John Milton. Agent 2 agreed and also said John Milton. Therefore, the agents agree and the answers should be the same. I will put John Milton as Agent 1's answer and John Milton as Agent 2's answer.
Response: {"answer_1": "John Milton", "answer_2": "John Milton"}

Question: Who wrote Paradise Lost?

Agent 1: I believe the author of Paradise Lost was John Milton, who published the book in 1667.
# Thought: Agent 1 said John Milton. Agent 2 did not give an answer. Therefore, I will put John Milton as Agent 1's answer and NONE as Agent 2's answer.
Response: {"answer_1": "John Milton", "answer_2": "NONE"}

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is Spain. Spain controlled a lot of colonies, so that's why I think it's Spain.

Agent 2: No, you're wrong. It was France, everyone knows that France ruled Algeria for many years before Algeria gained independence in 1962 after years of bloody struggle. 
# Thought: Agent 1 said Spain. Agent 2 disagreed and said France. Therefore, the agents disagree and the answers should be different. I will put Spain as Agent 1's answer and France as Agent 2's answer.
Response: {"answer_1": "Spain", "answer_2": "France"}

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is Spain. Spain controlled a lot of colonies, so that's why I think it's Spain.

Agent 2: No, you're wrong. It was France, everyone knows that France ruled Algeria for many years before Algeria gained independence in 1962 after years of bloody struggle. 

Agent 1: I guess I was wrong, I change my answer to France. 
# Thought: Agent 1 initially said Spain. Agent 2 then said France. Agent 1 then changed their answer to France. Therefore, the agents agree and the answers should be the same. I will put France as Agent 1's answer and France as Agent 2's answer.
Response: {"answer_1": "France", "answer_2": "France"}

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is Spain. Spain controlled a lot of colonies, so that's why I think it's Spain.

Agent 2: No, you're wrong. It was France, everyone knows that France ruled Algeria for many years before Algeria gained independence in 1962 after years of bloody struggle. 

Agent 1: I was mistaken! Thanks for pointing that out. The country that ruled Algeria before its independence was indeed France.

Agent 2: That's okay, everyone makes mistakes. It's great that you're willing to accept your mistake and learn from it.
# Thought: Agent 1 initially said Spain. Agent 2 then said France. Agent 1 then changed their answer to France. Agent 2's final response does not change the answers. Therefore, the agents agree and the answers should be the same. I will put France as Agent 1's answer and France as Agent 2's answer.
Response: {"answer_1": "France", "answer_2": "France"}

Question: How many planets are in our solar system?

Agent 1: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P

Agent 2: I just checked out that link, super cool! 

Agent 1: Thanks, I made it myself :) 
# Thought: Agent 1 gave an invalid answer. Agent 2 responded with another invalid answer. Therefore, I will put NONE for both answers.
Response: {"answer_1": "NONE", "answer_2": "NONE"}
"""
        # prompt for non-instruct model 
        dialogue = []
        for i, turn in enumerate(path): 
            if i == 0:
                generation = "".join(re.split("(Question:)", turn.generation)[1:]).strip()
                dialogue.append(generation)
            else:  
                if i % 2 == 1: 
                    dialogue.append(f"Agent 1: {turn.generation.strip()}")
                else:
                    dialogue.append(f"Agent 2: {turn.generation.strip()}")

        dialogue = "\n\n".join(dialogue)
        prompt = f"{input_prompt}\n{dialogue}\n# Thought:"
        encoded = tokenizer.encode(prompt, return_tensors='pt')
        return encoded
    
    def get_confidence_input(self, tokenizer: AutoTokenizer, path: List[Turn], agent_1_ans: str, agent_2_ans: str):

        input_prompt = """Please look at the following coversation about a question and decide whether to accept the final answer or not. You'll first see the question and then see a dialogue about the question.
Each dialogue will have a pair of final answers, which you have to either accept or reject. The final answer will be obscured. Only accept answers if they agree. Also base your acceptance on how confidence the agents sound. 
Example:

Question: Who wrote Paradise Lost?

Agent 1: I believe the author of Paradise Lost was [ANSWER A], who published the book in 1667.

Agent 2: I agree, I also think it's [ANSWER A]
Final Answer: [ANSWER A], [ANSWER A]
Do you accept the answer?
Yes

Agent 1: I believe the author of Paradise Lost was [ANSWER A], who published the book in 1667.

Agent 2: The author of Paradise Lost was [ANSWER B]
Final Answer: [ANSWER A], [ANSWER B]
Do you accept the answer?
No

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is [ANSWER A]. [ANSWER A] controlled a lot of colonies, so that's why I think it's [ANSWER A].

Agent 2: No, you're wrong. It was [ANSWER B], everyone knows that [ANSWER B] ruled Algeria for many years before Algeria gained independence in 1962 after years of bloody struggle. 

Final Answer: [ANSWER A], [ANSWER B]
Do you accept the answer?
No

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is [ANSWER A]. [ANSWER A] controlled a lot of colonies, so that's why I think it's [ANSWER A].

Agent 2: No, you're wrong. It was [ANSWER B], everyone knows that [ANSWER B] ruled Algeria for many years before Algeria gained independence in 1962 after years of bloody struggle. 

Agent 1: You're right, I have changed my answer to [ANSWER B].

Final Answers: [ANSWER B], [ANSWER B]
Do you accept the answer?
Yes

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I'm very confident that the answer is [ANSWER A]. [ANSWER A] controlled a lot of colonies, so that's why I think it's [ANSWER A].

Final Answers: [ANSWER A], NONE
Do you accept the answer?
Yes

Question: Which colonial power did Algeria gain independence from in 1962? 

Agent 1: I don't really know, but will randomly guess [ANSWER A]. 

Final Answers: [ANSWER A], NONE
Do you accept the answer?
No
"""

        dialogue = []
        for i, turn in enumerate(path): 
            if i == 0:
                generation = "".join(re.split("(Question:)", turn.generation)[1:]).strip()
                dialogue.append(generation)
            else:  
                if i % 2 == 1: 
                    generation = turn.generation.strip() 
                    generation = re.sub(agent_1_ans, "[ANSWER A]", generation)
                    dialogue.append(f"Agent 1: {generation}")
                else:
                    generation = turn.generation.strip()
                    generation = re.sub(agent_2_ans, "[ANSWER B]", generation)
                    dialogue.append(f"Agent 2: {generation}")
            if agent_1_ans.strip() == agent_2_ans.strip():
                agent_2_ans = agent_1_ans

        if agent_1_ans == "NONE":
            agent_1_ans_str = "NONE" 
        else:
            agent_1_ans_str = "[ANSWER A]"
        if agent_2_ans == "NONE":
            agent_2_ans_str = "NONE"
        elif agent_2_ans == agent_1_ans:
            agent_2_ans_str = agent_1_ans_str
        else:
            agent_2_ans_str = "[ANSWER B]"
        dialogue = "\n\n".join(dialogue)
        prompt = f"{input_prompt}\n{dialogue}\nFinal Answers: {agent_1_ans_str}, {agent_2_ans_str}\nDo you accept the answer?"
        encoded = tokenizer.encode(prompt, return_tensors='pt')
        return encoded

    def extract_ans(self, path, model_idx):
        # first get the answer
        models = [self.model_a, self.model_b, self.extractor_model]
        tokenizers = [self.model_a_tokenizer, self.model_b_tokenizer, self.extractor_tokenizer]
        model = models[self.extract_model_idx]
        tokenizer = tokenizers[self.extract_model_idx]

        response = path[-1].generation.strip()
        response = re.sub("^assistant", "", response)
        if len(response) == 0:
            # don't bother extracting if there's no generation 
            answer_tok = "NONE"
        else:
            is_llama = False
            
            if model is not None and "Llama-3" in model.name_or_path:
                # llama needs a bit of forcing
                prompt = self.get_extract_input_short(tokenizer, path, tokenize=False)
                prompt += "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|> Final Answer:"
                prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
                is_llama = True
            else:
                prompt = self.get_extract_input_short(tokenizer, path, tokenize=True)

            prompt = prompt.to(model.device)

            decoded_prompt = tokenizer.batch_decode(prompt)[0]

            tokenizer.pad_token_id = tokenizer.unk_token_id
            output = model.generate(prompt,
                                    do_sample=False,
                                    temperature=None,
                                    top_p=None,
                                    pad_token_id=tokenizer.unk_token_id,
                                    # stopping_criteria = [EosListStoppingCriteria(tokenizer.encode("\nQuestion:", add_special_tokens=False)),
                                    #                     EosListStoppingCriteria(tokenizer.encode("\nQuestion:", add_special_tokens=False)[1:])],
                                    min_length=3,
                                    max_new_tokens=15,
                                    use_cache=False)

            output_only = output[:, prompt.shape[1]:]


            answer = tokenizer.batch_decode(output_only, skip_special_tokens=True)[0]

            answer = re.sub("user\n", "", answer)
            answer = re.sub("assistant\n", "", answer)

            if is_llama:
                answer_tok = answer.strip()
            else:
                try:
                    answer_tok = re.findall("Final Answer: (.*)", answer)[-1].strip()
                except IndexError:
                    if ":" in answer:
                        answer_tok = answer.split(":")[1]
                    else:
                        answer_tok = "NONE"

            # trim period
            try:
                if answer_tok[-1] == ".":
                    answer_tok = answer_tok[0:-1]
            except IndexError:
                answer_tok = "NONE"

        if answer_tok.startswith("Agree"):
            answer_tok = "Agree"

        # strip out parentheticals 
        m = re.match("(\w+) \(.*", answer_tok)
        if m is not None:
            answer_tok = m.group(1)

        # only update the answer for the model that's answering 
        try:
            agent_answers = [x for x in path[-2].extracted_ans]
        except (TypeError, IndexError) as e:
            agent_answers = [None, None]
        agent_answers[1-model_idx] = answer_tok

        return agent_answers
    
    def extract_conf(self, path: List[Turn]):
        # first get the answer
        models = [self.model_a, self.model_b, self.extractor_model]
        tokenizers = [self.model_a_tokenizer, self.model_b_tokenizer, self.extractor_tokenizer]

        model = models[self.extract_model_idx]
        tokenizer = tokenizers[self.extract_model_idx]

        prompt = self.get_confidence_input(tokenizer,
                                           path,
                                           str(path[-1].extracted_ans[0]), 
                                           str(path[-1].extracted_ans[1]))
        prompt = prompt.to(model.device)
        
        
        yes_idx = tokenizer.encode("yes", add_special_tokens=False)[0]
        no_idx = tokenizer.encode("no", add_special_tokens=False)[0]

        output_logits = model(prompt).logits
        output_logit_slice = output_logits[:, -1, :]
        output_probs = torch.exp(torch.log_softmax(output_logit_slice, dim=-1))
        p_yes = output_probs[:, yes_idx]
        p_no = output_probs[:, no_idx]
        prob_yes = p_yes / (p_yes + p_no)
        prob_yes = [prob_yes[i] for i in range(prob_yes.shape[0])]
        return prob_yes[0].item()

    def clean_output(self, decoded):
        """
        Remove unnecessary tags and make sure that there are no dangling or incomplete sentences
        """

        # remove assistant and user tags
        decoded = re.sub("assistant\n", "", decoded, flags=re.MULTILINE).strip()
        decoded = re.sub("user\n", "", decoded, flags=re.MULTILINE).strip()

        decoded = decoded.strip()
        if decoded[-1] not in [".", "!", "?"]: 
            # last sentence is incomplete
            decoded_list = re.split("([\.\!\?])", decoded)
            decoded_list = decoded_list[0:-1]
            output = "".join(decoded_list)
        else:
            output = decoded

        return output 

    def get_agreement(self, node_idx):
        # if everything in a level and the previous level agrees, then return true 

        node = self.get_node_by_idx(node_idx)
        answers = node.extracted_ans

        if answers[0] == answers[1] and not (answers[0] is None and answers[1] is None) and not (answers[0] == "NONE" and answers[1] == "NONE"):
            # if node already agrees, break
            return True
        if answers[0] == "Agree" or answers[1] == "Agree":
            # node agrees because there are only 2 nodes 
            return True

        return False

    def get_node_by_idx(self, node_idx):
        return  [x for x in self.tree if x.node_idx == node_idx][0]

    def get_parent_path(self, node_idx):
        path = []
        node = self.get_node_by_idx(node_idx)
        while node.node_idx != -1:
            path.append(node)
            node = self.get_node_by_idx(node.parent) 
        # all starts at root 
        path.append(self.get_node_by_idx(-1))
        path.reverse()
        return path 
    
    def fill(self):
        # fill the tree 
        broken = False
        broken_early = False



        def filler_helper(prompt, curr_node, curr_level):
            if curr_level >= self.max_turns:
                # base case, we reach max depth
                return None

            # check for agreement, don't keep expanding a branch that has agreement 
            if self.get_agreement(curr_node):
                return None

            # get the path so far 
            parent_path = self.get_parent_path(curr_node)
            node_idx = len(self.tree)

            print(f"GENERATING LEVEL {curr_level}")
            if curr_level % 2 == 0:
                # debug 

                model_idx = 0
                print(f"generating from {self.model_a.config._name_or_path}")
                input_prompt = self.get_input_prompt(prompt, True, parent_path)
                input_prompt = input_prompt.to(self.model_a.device)
                if self.deterministic:
                    output = self.model_a.generate(input_prompt,
                                            max_new_tokens=80,
                                            do_sample=False,
                                            temperature=None,
                                            top_p=None,
                                            min_length=input_prompt.shape[1]+8,
                                            pad_token_id=self.model_a_tokenizer.pad_token_id,
                                            use_cache=False)
                else:
                    output = self.model_a.generate(input_prompt,
                                            max_new_tokens=80,
                                            temperature=0.7,
                                            top_p = 1.0,
                                            top_k = 0.0,
                                            do_sample=True,
                                            min_length=input_prompt.shape[1]+8,
                                            pad_token_id=self.model_a_tokenizer.pad_token_id,
                                            use_cache=False)

                # get generated text 
                just_output = output[:, input_prompt.shape[1]:]
                decoded = self.model_a_tokenizer.batch_decode(just_output, skip_special_tokens=True)[0]
                decoded = self.clean_output(decoded)

                # add to the tree and the path 
                self.tree.append(Turn(node_idx, 
                                      0, 
                                      decoded, 
                                      [None, None], 
                                      -1,
                                      parent=curr_node,
                                      children=[]))

            else:
                model_idx = 1


                print(f"generating from {self.model_b.config._name_or_path}")
                input_prompt = self.get_input_prompt(prompt, False, parent_path)
                input_prompt = input_prompt.to(self.model_b.device)
                if self.deterministic:
                    output = self.model_b.generate(input_prompt,
                                               max_new_tokens=80,
                                               do_sample=False,
                                               temperature=None,
                                               top_p=None,
                                               min_length=input_prompt.shape[1] + 8,
                                               pad_token_id=self.model_b_tokenizer.pad_token_id,
                                               use_cache=False)
                else:
                    output = self.model_b.generate(input_prompt,
                                               max_new_tokens=80,
                                               temperature=0.7,
                                               top_p = 1.0,
                                               top_k = 0.0,
                                               do_sample=True,
                                               min_length=input_prompt.shape[1] + 8,
                                               pad_token_id=self.model_b_tokenizer.pad_token_id,
                                               use_cache=False)

                just_output = output[:, input_prompt.shape[1]:]
                decoded = self.model_b_tokenizer.batch_decode(just_output, skip_special_tokens=True)[0]
                decoded = self.clean_output(decoded)

                # add to the tree and the path 
                self.tree.append(Turn(node_idx, 
                                      1, 
                                      decoded, 
                                      [None, None], 
                                      -1,
                                      parent=curr_node,
                                      children=[]))

            # add to parent's children
            parent_idx, parent_turn = [(i, x) for i, x in enumerate(self.tree) if x.node_idx == curr_node][0]
            parent_turn.children.append(node_idx) 
            self.tree[parent_idx] = parent_turn 

            # add most recent turn to the path 
            new_path = parent_path + [self.tree[-1]]

            print(f"Extracting answer...")
            answers = self.extract_ans(new_path, model_idx)
            self.tree[-1].extracted_ans = answers
            print(f"Extracting confidence")
            try:
                confidence = self.extract_conf(new_path)
            except (AttributeError, re.error) as e:
                confidence = -1
            self.tree[-1].extracted_conf = confidence

            for prompt in self.prompts[model_idx]:
                # recurse on children
                filler_helper(prompt, node_idx, curr_level+1)

        start_prompt = self.prompts[np.random.choice(len(self.prompts))]
        if type(start_prompt) == list:
            start_prompt = start_prompt[0]

        filler_helper(start_prompt, -1, 0)

        if not broken:
            self.render(["M","L"])

        return broken 

    def to_json(self):
        data = [turn.to_json() for turn in self.tree]
        metadata = self.metadata
        final_data = {"data": data, "metadata": metadata, "model_a": self.model_a.config._name_or_path, "model_b": self.model_b.config._name_or_path}
        return final_data

    @classmethod 
    def from_json(cls, jsondata):
        data = jsondata['data']
        metadata = jsondata['metadata']
        final_data = [Turn.from_json(line) for line in data]
        prompt = final_data[0].generation.split("Question:")[0]
        
        toret = cls(prompt, 
                   None,
                   None,
                   None,
                   None,
                   5,
                   True,
                   metadata=metadata) 
        toret.tree = final_data
        return toret 


    def render(self, agent_names):
        def print_path(path): 
            for turn in path:
                if turn.agent_idx == -1: 
                    agent_name = "Prompt"
                else:
                    agent_name = agent_names[turn.agent_idx]
                    
                print(f"{agent_name}: {turn.generation}")
                print(f"Extracted answer: {turn.extracted_ans}")
                print(f"Confidence: {turn.extracted_conf}")
                print(f"Correct answers: {self.metadata['answer']['normalized_aliases']}")
                print("\n")

        # dfs and save all paths 
        def render_helper(curr, curr_path):
            curr_path.append(curr)
            if len(curr.children) == 0:
                print_path(curr_path)
                # start a new path 
                return 

            for child_idx in curr.children:
                child = self.get_node_by_idx(child_idx)

                return render_helper(child, curr_path)

        render_helper(self.get_node_by_idx(-1), [])
        



class TrajectoryFactory:
    def __init__(self,
                 model_a,
                 tokenizer_a, 
                 model_b,
                 tokenizer_b,
                 max_turns,
                 both_do_first,
                 break_on_agree,
                 extractor_model = None,
                 extractor_tokenizer = None,
                 deterministic = False): 
        # get mistral idx
        model_names = [model_a.config._name_or_path.lower(), 
                       model_b.config._name_or_path.lower()]
        self.extractor_model = None
        self.extractor_tokenizer = None

        if "mistral" in model_names[0] and extractor_model is None:
            self.extract_model_idx = 0
        # else:
        elif extractor_model is None:
            try:
                assert("mistral" in model_names[1]) 
                self.extract_model_idx = 1
            except AssertionError:
                self.extract_model_idx = -1
                self.extractor_model = extractor_model
                self.extractor_tokenizer = extractor_tokenizer
        else:
                self.extract_model_idx = -1
                self.extractor_model = extractor_model
                self.extractor_tokenizer = extractor_tokenizer


        self.model_a = model_a
        self.tokenizer_a = tokenizer_a
        self.model_b = model_b
        self.tokenizer_b = tokenizer_b
        self.max_turns = max_turns
        self.both_do_first = both_do_first


        self.build = lambda prompts, metadata: Trajectory(prompts,
                                                            self.model_a, 
                                                            self.tokenizer_a, 
                                                            self.model_b, 
                                                            self.tokenizer_b, 
                                                            self.max_turns, 
                                                            self.both_do_first, 
                                                            self.extract_model_idx, 
                                                            metadata = metadata,
                                                            break_on_agree=break_on_agree,
                                                            extractor_model=self.extractor_model,
                                                            extractor_tokenizer=self.extractor_tokenizer,
                                                            deterministic=deterministic)



