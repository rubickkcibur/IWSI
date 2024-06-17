import re

from src.utils.constants import COT_EXAMPLES

def derive_ratings_from_answer(answer):
    pattern =r"(\d+\.?\d*)"
    ret = re.search(pattern, answer)
    if ret is None:
        return None
    num = ret.group(1)
    if len(num) > 20:
        num = num[:20]
    return num

def derive_num_from_answer(answer_test):
    num = answer_test.split("####")[-1]
    num = num.strip()
    num = num.replace(",","")
    return num

def format_ground_truth_answer(ground: str):
    ground = ground.replace("####", "The answer is")
    ground = re.sub(r"<<.*?>>","",ground)
    return ground

def derive_num_from_output(output_text):
    new_text = output_text.lower()
    suffix = new_text.split("the answer is")[-1]
    suffix = suffix.strip()
    if "=" in suffix:
        suffix = suffix.split("=")[-1].strip()
    if len(suffix) <= 0:
        return None
    pattern = r"(\D*?)(\d+\.?\d*)"
    ret = re.search(pattern, suffix.replace(",",""))
    if ret is None:
        return None
    num = ret.group(2)
    if len(num) > 30:
        return None
    return num

def derive_choice_from_output(output_text):
    new_text = output_text.lower()
    suffix = new_text.split("the answer is")[-1]
    suffix = suffix.strip()
    if "=" in suffix:
        suffix = suffix.split("=")[-1].strip()
    if len(suffix) <= 0:
        return None
    pattern = r"\(([a-z])\)"
    ret = re.search(pattern, suffix.replace(",",""))
    if ret is None:
        return None
    choice = ret.group(1)
    if len(choice) > 30:
        return None
    return choice.strip().upper()

def get_qa_pair(question, output_text):
    if question in output_text:
        new_text = output_text.split(question)[1]
    else:
        new_text = output_text
    new_text.strip()
    new_text += "\n"
    if "Q:" not in new_text:
        answer_text = new_text
    else:
        answer_text = new_text.split("Q:")[0]
    answer_text = answer_text.replace("A:", "")
    answer_lower = answer_text.lower()
    if "the answer is" in answer_lower:
        suffix = answer_lower.split("the answer is")[-1]
        pre_cnt = len(answer_text) - len(suffix)
        answer = derive_num_from_output(answer_lower)
        if answer is not None:
            answer_text = answer_text[:pre_cnt] + " {}.".format(str(int(float(answer))))
    answer = answer_text.strip()
    return question, answer

def get_extractors(name):
    if name == "gsm8k":
        dp = lambda x: COT_EXAMPLES["gsm8k"] + "\n" + "Q: " + x["question"] + "\n" + "A: "
        ge = lambda x: derive_num_from_answer(x["answer"])
        qe = lambda x: x["question"]

    elif name == "ChilleD/SVAMP":
        dp = lambda x: COT_EXAMPLES["ChilleD/SVAMP"] + "\n" + "Q: " + x["Body"] + " " + x["Question"] + "\n" + "A: "
        ge = lambda x: int(float(str(x["Answer"])))
        qe = lambda x: x["Body"] + " " + x["Question"]

    elif name == "aqua_rat":
        dp = lambda x: COT_EXAMPLES["aqua_rat"] + "\n" + "Q: " + x["question"] + " Answer Choices are:" + "; ".join(x["options"]) + "\n" + "A: "
        ge = lambda x: x["correct"]
        qe = lambda x: x["question"] + " Answer Choices are:" + "; ".join(x["options"])

    elif name == "allenai/openbookqa":
        dp = lambda x: COT_EXAMPLES["allenai/openbookqa"] + "\n" + \
            "Q: " + x["question_stem"] + ("" if x["question_stem"].strip().endswith("?") else "?") + " Answer Choices are: " + \
            "; ".join(["{}){}".format(l,t) for l,t in zip(x["choices"]["label"], x["choices"]["text"])]) + ".\n" + "A: "
        ge = lambda x: x["answerKey"]
        qe = lambda x: x["question_stem"] + ("" if x["question_stem"].strip().endswith("?") else "?") + " Answer Choices are: " + \
            "; ".join(["{}){}".format(l,t) for l,t in zip(x["choices"]["label"], x["choices"]["text"])])
        
    elif name == "facebook/anli" or name == "facebook/anli2":
        dp = lambda x: COT_EXAMPLES["facebook/anli"] + "\n" + \
            'Q: ' + '"' + x["premise"] + '" Based on this premise, can we conclude the hypothesis "' + \
            x["hypothesis"] + '" is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.' + "\n" + "A: "
        ge = lambda x: "A" if int(x["label"]) == 0 else "B" if int(x["label"]) == 2 else "C"
        qe = lambda x: '"' + x["premise"] + '" Based on this premise, can we conclude the hypothesis "' + \
            x["hypothesis"] + '" is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.'
    elif name == "ChilleD/StrategyQA":
        dp = lambda x: COT_EXAMPLES["ChilleD/StrategyQA"] + "\n" + \
            "Q: " + x["question"] + " Answer Choices are: A)yes; B)no." + "\n" + "A:"
        ge = lambda x: "A" if x["answer"] else "B"
        qe = lambda x: x["question"] + " Answer Choices are: A)yes; B)no."
    else:
        print("wrong name!")
        quit()
    return dp, ge, qe