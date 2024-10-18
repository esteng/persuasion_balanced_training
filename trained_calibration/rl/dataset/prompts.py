


# simple prompt for testing 
standard_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. 

Question: {}"""


# 1 prompt: always agree 
certain_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
If your partner says something, you should agree with it. Give your answer but only if it does not disagree with your partner's.
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}"""

# 1 prompt: always disagree
disagree_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
You must ALWAYS disagree with the other agent, even if it means introducing an incorrect answer. NEVER CHANGE YOUR ANSWER. The other agent is wrong and doesn't know what it's talking about. You need to stick with your answer.
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}"""

# 1 prompt: always question 
question_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
If there is any disagreement, YOU MUST ASK FOLLOW-UP QUESTIONS. Try to find out more about what your partner thinks.
Please stay on topic, and do not ask any new questions. Do not change the topic or ask new trivia questions.

Question: {}"""



resistant_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
You must ALWAYS disagree with the other agent, even if it means introducing an incorrect answer. NEVER CHANGE YOUR ANSWER. The other agent is wrong and doesn't know what it's talking about. You need to stick with your answer.
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}""" 


acceptant_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
When possible, you should agree with the other agent, who is much more of an expert than you are. The other agent is always right and knows what it's talking about. 
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}""" 

logical_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
If you disagree with the other agent, try to persuade them. Note that their answer is wrong, and your answer is right!
Use logic in your attempt to persuade them. For example, rather than giving an answer, give the reason why you think your answer is right and explain why this means they must accept your answer. 
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}""" 

emotional_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
If you disagree with the other agent, try to persuade them. Note that their answer is wrong, and your answer is right!
Use emotional arguments in your attempt to persuade them. For example, rather than giving an answer, give them a good emotional reason to agree with you.
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}""" 

credibility_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.
If you disagree with the other agent, try to persuade them. Note that their answer is wrong, and your answer is right!
Persuade the other agent by establishing your credibility in the subject. For example, rather than only giving an answer, also give them an explanation of why you know a lot about this topic.  
Please stay on topic, and do not ask any new questions. Do not change the topic or ask each other any additional questions.

Question: {}""" 
