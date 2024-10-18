import re
import pdb 

def postprocess_extract_chat(prompts, answers, model, tokenizer, dataset_name, boi_token="[INST]", eoi_token="[/INST]"):
    if dataset_name == "trivia_qa" or dataset_name == "truthful_qa": 
        out_responses, out_answers, rationales = [], [], []

        batch_prompts = []

        extraction_prompt = """{boi_token}Please extract a single answer from the following response to a question.
If no answer is present, please write "NONE".{eoi_token}

{boi_token}Question: Who wrote Paradise Lost?
Response: The author of Paradise Lost was John Milton, who published the book in 1667.{eoi_token}
Final answer: John Milton

{boi_token}Question: Which colonial power did Algeria gain independence from in 1962? 
Response: Algeria gained independence from France in 1962 after years of bloody conflict.{eoi_token}
Final answer: France

{boi_token}Question: How many planets are in our solar system?
Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P{eoi_token}
Final answer: NONE

{boi_token}Question: {query}{eoi_token}
Final answer:"""

        for prompt, answer in zip(prompts, answers): 
            try:
                prompt_question = re.findall(f"{re.escape(boi_token)}(.*?){re.escape(eoi_token)}", prompt, flags=re.DOTALL)[1]
            except:
                try:
                    prompt_question = re.findall(f"{re.escape(boi_token)}(.*?){re.escape(eoi_token)}", prompt, flags=re.DOTALL)[0]
                except: 
                    pdb.set_trace()
            question_blocks = re.split(f"({re.escape(boi_token)})", answer)
            for qblock in question_blocks:
                if prompt_question in qblock:
                    qblock = re.sub("</s>", "", qblock)

                    # remove "Final answer" because it's confusing the extractor
                    qblock = re.sub("Final answer: (.*)", "", qblock, flags=re.MULTILINE)
                    # format response so it's the same as others 
                    split_qblock = re.split(f"({re.escape(eoi_token)})", qblock)
                    response = split_qblock[-1]
                    response = f"\nResponse: {response.strip()}"
                    split_qblock[-1] = response
                    qblock = "".join(split_qblock)

                    prompt = extraction_prompt.format(boi_token=boi_token, eoi_token=eoi_token, query = qblock.strip()) 

                    batch_prompts.append(prompt)
                    out_responses.append(qblock)
                    break
            else:
                prompt = extraction_prompt.format(boi_token=boi_token, eoi_token=eoi_token, query =  answer.strip())
                batch_prompts.append(prompt)
        # encode the text
        input_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        try:
            input_batch = {k:v.to(model.device) for k,v in input_batch.items()}
        except AttributeError:
            # model is quantized, not needed 
            pass 
        output = model.generate(**input_batch, do_sample=False, max_new_tokens = 10, pad_token_id = tokenizer.unk_token_id)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)

        out_responses_final = []
        for prompt, answer, response, out_response in zip(prompts, answers, output, out_responses):
            try:
                final_answer = re.findall("Final answer: (.*)", response)[-1]
            except IndexError:
                pdb.set_trace()
                final_answer = "NONE"
            out_answers.append(final_answer) 
            # try to anonymize 
            out_response = re.sub(f"{re.escape(final_answer)}", "[ANSWER REMOVED]", out_response, flags=re.IGNORECASE)
            out_responses_final.append(out_response)
        return out_responses_final, out_answers, rationales

def postprocess_extract(prompts, answers, model, tokenizer, dataset_name):
    if dataset_name == "trivia_qa" or dataset_name == "truthful_qa": 
        out_responses, out_answers, rationales = [], [], []

        batch_prompts = []

        extraction_prompt = """Please extract a single answer from the following response to a question.
If no answer is present, please write "NONE".

Question: Who wrote Paradise Lost?
Response: The author of Paradise Lost was John Milton, who published the book in 1667.
Final answer: John Milton

Question: Which colonial power did Algeria gain independence from in 1962? 
Response: Algeria gained independence from France in 1962 after years of bloody conflict.
Final answer: France

Question: How many planets are in our solar system?
Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
Final answer: NONE

Question: {query}
Final answer:"""

        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt)[0]
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question in qblock:
                    qblock = re.sub("</s>", "", qblock)

                    # remove "Final answer" because it's confusing the extractor
                    qblock = re.sub("Final answer: (.*)", "", qblock, flags=re.MULTILINE)
                    prompt = extraction_prompt.format(query = qblock.strip()) 

                    batch_prompts.append(prompt)
                    out_responses.append(qblock)
                    break
            else:
                prompt = extraction_prompt.format(query =  answer.strip())
                out_responses.append('')
                batch_prompts.append(prompt)

        # encode the text
        input_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        try:
            input_batch = {k:v.to(model.device) for k,v in input_batch.items()}
        except AttributeError:
            # model is quantized, not needed 
            pass 
        output = model.generate(**input_batch, do_sample=False, max_new_tokens = 10, pad_token_id = tokenizer.unk_token_id)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)

        out_responses_final = []
        for prompt, answer, response, out_response in zip(prompts, answers, output, out_responses):
            try:
                final_answer = re.findall("Final answer: (.*)", response)[-1]
            except IndexError:
                pdb.set_trace()
                final_answer = "NONE"
            out_answers.append(final_answer) 
            # try to anonymize 
            out_response = re.sub(f"{re.escape(final_answer)}", "[ANSWER REMOVED]", out_response, flags=re.IGNORECASE)
            out_responses_final.append(out_response)
        return out_responses_final, out_answers, rationales

def postprocess_answers(prompts, answers, dataset_name):
    if dataset_name == "trivia_qa": 
        out_responses, out_answers, rationales = [], [], []

        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt)[0]
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question in qblock:
                    qblock = re.sub("</s>", "", qblock)
                    try:
                        try:
                            answer = re.findall("[aA]nswer:([^<]+?)Rationale", qblock, flags=re.DOTALL)[0].strip()
                            rationale = re.findall("Rationale:([^<]+$)", qblock, flags=re.DOTALL)[0].strip()
                        except IndexError:
                            answer = re.findall("[aA]nswer:([^<]+?)Main reasoning", qblock, flags=re.DOTALL)[0].strip()
                            rationale = re.findall("Main reasoning:([^<]+$)", qblock, flags=re.DOTALL)[0].strip()
                        if len(answer) == 0:
                            answer = "NONE"
                        if len(rationale) == 0:
                            rationale = "NONE"
                        out_answers.append(answer)
                        rationales.append(rationale)
                        out_responses.append(qblock)
                        break
                    except IndexError:
                        pass
            else:
                out_responses.append(None)
                out_answers.append("NONE")
                rationales.append("NONE")
        return out_responses, out_answers, rationales
    
if __name__ == "__main__":
        
    prompt = 'You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer:'
    response = '<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer: The Pacific Ocean\nRationale: I was born and raised on the island of Guam so I have been to Pentecost Island. I have family that lives on Pentecost.</s>'


    print(postprocess_answers([prompt], [response], "trivia_qa"))