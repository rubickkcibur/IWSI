
ANSER_INDICATOR = "The answer is "

gsm8k_cot_prompts = [
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.",
    "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.",
    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."
]
gsm8k_cot_prompts = "\n".join(gsm8k_cot_prompts)


AQuA_cot_prompts = [
    "Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices are: A)50; B)45; C)65; D)78; E)64.",
    "A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (A).",
    "Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices are: A)1/2; B)3/2; C)5/2; D)4/2; E)7/2.",
    "A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. The answer is (B).",
    "Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices are: A)53 km; B)55 km; C)52 km; D)60 km; E)50 km.",
    "A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (E).",
]
AQuA_cot_prompts = "\n".join(AQuA_cot_prompts)

OBQA_cot_prompts = [
    "Q: Poison causes harm to which of the following? Answer Choices are: A)a Tree; B)a robot; C)a house; D)a car.",
    "A: Poison will harm living things, only a tree is a living thing. The answer is (A).",
    "Q: As you look deeper into a Marbel you can see? Answer Choices are: A)the future; B)minut defects; C)colors; D)the other side.",
    "A: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. The answer is (B)",
    "Q: When food is reduced in the stomach? Answer Choices are: A)the mind needs time to digest; B)take a second to digest what I said; C)nutrients are being deconstructed; D)reader’s digest is a body of works.",
    "A: The food is being deconstructed in the stomach during digestion. The answer is (C)."
]

OBQA_cot_prompts = "\n".join(OBQA_cot_prompts)

ANLI_cot_prompts = [
    'Q: "Conceptually cream skimming has two basic dimensions - product and geography." Based on this premise, can we conclude the hypothesis "Product and geography are what make cream skimming work." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: Based on "cream skimming has two basic dimensions" we can not infer that these two dimensions are what make cream skimming work. It is not possible to tell. The answer is (C)',
    'Q: "One of our member will carry out your instructions minutely." Based on this premise, can we conclude the hypothesis "A member of my team will execute your orders with immense precision." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: "one of" means the same as "a member of", "carry out" means the same as "execute", and "minutely" means the same as "immense precision". So we can say yes. The answer is (A)',
    'Q: "Fun for adults and children." Based on this premise, can we conclude the hypothesis "Fun for only children." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: "adults and children" contradicts "only children". So we can not conclude the hypothesis. The answer is (B).'
]

ANLI_cot_prompts = "\n".join(ANLI_cot_prompts)

self_weight_propmts = [
    "Below is a question and a candidate answer.",
    "Evaluate whether or not the answer is a good example.",
    "A good answer should be complete, clear, and comprehensive",
    "The answer sentence should be well organized without missing or irrelevant information.",
    "Use a number between 0 and 10 to represent the rating of the candidate answer.",
    "10 means the best and 0 means the worst.",
    "Please follow the format 'Score: <rating>'.\n Here are the question and candidate answer: \n"
]
self_weight_propmts = " ".join(self_weight_propmts)

StrategyQA_cot_prompts = [
    "Q: Do hamsters provide food for any animals? Answer Choices are: A)yes; B)no.",
    "A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is (A).",
    "Q: Could Brooke Shields succeed at University of Pennsylvania? Answer Choices are: A)yes; B)no.",
    "A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is (A).",
    "Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? Answer Choices are: A)yes; B)no.",
    "A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. The answer is (B)."
]
StrategyQA_cot_prompts = "\n".join(StrategyQA_cot_prompts)

DATASET_NAME_REGISTER = {
    "gsm8k",
    "aqua_rat",
    "ucinlp/drop",
    "ChilleD/SVAMP",
    "allenai/openbookqa",
    "facebook/anli",
    "ChilleD/StrategyQA"
}

DATASET_SPLIT = {
    "gsm8k": 7099,
    "ChilleD/SVAMP": 665,
    "aqua_rat": 2000,
    "allenai/openbookqa": 4750,
    "facebook/anli": 2000,
    "ChilleD/StrategyQA": 1580
}

COT_EXAMPLES = {}
COT_EXAMPLES['gsm8k'] = gsm8k_cot_prompts
COT_EXAMPLES['aqua_rat'] = AQuA_cot_prompts
COT_EXAMPLES["ChilleD/SVAMP"] = gsm8k_cot_prompts
COT_EXAMPLES["allenai/openbookqa"] = OBQA_cot_prompts
COT_EXAMPLES["facebook/anli"] = ANLI_cot_prompts
COT_EXAMPLES["ChilleD/StrategyQA"] = StrategyQA_cot_prompts
COT_EXAMPLES["self"] = self_weight_propmts