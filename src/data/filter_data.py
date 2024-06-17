import jsonlines
import random
from src.utils.utils import derive_num_from_answer, derive_num_from_output, derive_choice_from_output
from src.model.filterLM import FilterModel
import os
import math
import json
import numpy as np


def filter_data(question, candidates, ground, mode="Consistency", uncertainty_threshold=1, model=None, num=True):
    assert mode in ["Consistency", "Groundtruth", "Entropy", "Weighted", "None"]
    assert not (mode == "Weighted" and model is None)
    num_candidates = len(candidates)
    if mode == "None":
        marks = [1] * num_candidates
        return marks
    if mode == "Weighted":
        batch_inputs = ["Q: " + question + "\n" + "A: " + cand for cand in candidates]
        weights = model.predict(batch_inputs)
        return list(weights)
    candidate_answers = [derive_num_from_output(cand) for cand in candidates] if num else [derive_choice_from_output(cand) for cand in candidates]
    if num:
        candidate_answers = [int(float(ca)) if ca is not None  else None for ca in candidate_answers]
    stat = {}
    for id, ca in enumerate(candidate_answers):
        if ca is None:
            continue
        if ca not in stat:
            stat[ca] = []
        stat[ca].append(id)
    max_ca = None
    if mode == "Consistency":
        max_num = 0
        for k,v in stat.items():
            if len(v) > max_num:
                max_ca = k
                max_num = len(v)
    elif mode == "Groundtruth":
        if num:
            for k in stat.keys():
                if int(float(k)) == int(float(ground)):
                    max_ca = k
        else:
            for k in stat.keys():
                if k == ground:
                    max_ca = k
    elif mode == "Entropy":
        max_num = 0
        probs = []
        for k,v in stat.items():
            probs.append(len(v))
            if len(v) > max_num:
                max_ca = k
                max_num = len(v)
        if len(probs) == 0:
            return None
        total = sum(probs)
        probs = [p/total for p in probs]
        entropy = sum([-p*math.log(p) for p in probs])
        max_en = math.log(total)
        if max_en == 0:
            return None
        if entropy / max_en > uncertainty_threshold:
            return None
    if max_ca is None:
        return None
    marks = [0] * num_candidates
    for i in stat[max_ca]:
        marks[i] = 1
    return marks

def get_data_weight(data_args, model=None):
    jsonl_path = data_args.data_path
    mode = data_args.data_filter_mode
    uncertainty_threshold = data_args.uncertainty_th
    temp_data_path = data_args.temp_data_path
    answer_is_num = data_args.dataset_name in ["gsm8k", "ChilleD/SVAMP"]
    weights = []
    if data_args.dataset_name is None or data_args.dataset_name == "":
        weights_path = os.path.join(temp_data_path, "weight.json")
        RM_weights_path = os.path.join(temp_data_path, "RM_weight.json")
        self_weights_path = os.path.join(temp_data_path, "self_weight.json")
    else:
        weights_path = os.path.join(temp_data_path, data_args.dataset_name.replace("/","_"), "weight.json")
        RM_weights_path = os.path.join(temp_data_path, data_args.dataset_name.replace("/","_"), "RM_weight.json")
        self_weights_path = os.path.join(temp_data_path, data_args.dataset_name.replace("/","_"), "self_weight.json")
    if mode == "Weighted":
        assert temp_data_path is not None and len(temp_data_path)>0
        with open(weights_path,"r",encoding="utf8") as f:
            weights = json.load(f)
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = max(weights[i][j], 2)
    elif mode == "Mixed":
        assert temp_data_path is not None and len(temp_data_path)>0
        with open(weights_path,"r",encoding="utf8") as f:
            DIWweights = json.load(f)
        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                question = obj["question"]
                candidates = obj["candidates"]
                ground = obj["ground_truth"]
                filter_marks = filter_data(question, candidates, ground, "Consistency", uncertainty_threshold, model, answer_is_num)
                if filter_marks is None:
                    filter_marks = [0] * len(candidates)
                weights.append(filter_marks)
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j] * max(DIWweights[i][j],2)
    elif mode == "K-Mixed":
        assert temp_data_path is not None and len(temp_data_path)>0
        with open(weights_path,"r",encoding="utf8") as f:
            DIWweights = json.load(f)
        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                question = obj["question"]
                candidates = obj["candidates"]
                ground = obj["ground_truth"]
                filter_marks = filter_data(question, candidates, ground, "Consistency", uncertainty_threshold, model, answer_is_num)
                if filter_marks is None:
                    filter_marks = [0] * len(candidates)
                weights.append(filter_marks)
        loss_diff = []
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j] * DIWweights[i][j]
                if weights[i][j] > 0:
                    # loss_diff.append(math.fabs(weights[i][j] - 1))
                    loss_diff.append(weights[i][j] if weights[i][j] > 1 else min(1/weights[i][j],5))
        loss_diff.sort()
        k_portion = int(len(loss_diff) * uncertainty_threshold)
        diff_th = loss_diff[k_portion]
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if weights[i][j] == 0:
                    continue
                diff = weights[i][j] if weights[i][j] > 1 else min(1/weights[i][j],5)
                weights[i][j] = 1 if diff < diff_th else 0
    
    elif mode == "RM":
        assert temp_data_path is not None and len(temp_data_path)>0
        with open(RM_weights_path,"r",encoding="utf8") as f:
            RM_weight = json.load(f)
        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                question = obj["question"]
                candidates = obj["candidates"]
                ground = obj["ground_truth"]
                filter_marks = filter_data(question, candidates, ground, "Consistency", uncertainty_threshold, model, answer_is_num)
                if filter_marks is None:
                    filter_marks = [0] * len(candidates)
                weights.append(filter_marks)
        loss_diff = []
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j] * RM_weight[i][j]
                if weights[i][j] != 0:
                    loss_diff.append(weights[i][j])
        loss_diff.sort(reverse=True)
        k_portion = int(len(loss_diff) * uncertainty_threshold)
        diff_th = loss_diff[k_portion]
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if weights[i][j] == 0:
                    continue
                diff = weights[i][j]
                weights[i][j] = 0 if diff < diff_th else 1
    elif mode == "Self":
        assert temp_data_path is not None and len(temp_data_path)>0
        with open(self_weights_path,"r",encoding="utf8") as f:
            self_weight = json.load(f)
        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                question = obj["question"]
                candidates = obj["candidates"]
                ground = obj["ground_truth"]
                filter_marks = filter_data(question, candidates, ground, "Consistency", uncertainty_threshold, model, answer_is_num)
                if filter_marks is None:
                    filter_marks = [0] * len(candidates)
                weights.append(filter_marks)
        loss_diff = []
        total = np.sum(np.array(weights))
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if weights[i][j] != 0:
                    loss_diff.append((self_weight[i][j],(i,j)))
                weights[i][j] = weights[i][j] * self_weight[i][j]
        loss_diff.sort(key=lambda x:x[0], reverse=True)
        k_portion = int(len(loss_diff) * uncertainty_threshold)
        diff_th = loss_diff[k_portion][0]
        if diff_th > 0:
            for i in range(len(weights)):
                for j in range(len(weights[i])):
                    if weights[i][j] == 0:
                        continue
                    diff = weights[i][j]
                    weights[i][j] = 0 if diff < diff_th else 1
        else:
            for item in loss_diff[:k_portion]:
                loca = item[1]
                weights[loca[0]][loca[1]] = 1
            for item in loss_diff[k_portion:]:
                loca = item[1]
                weights[loca[0]][loca[1]] = 0
        print("portion is ", np.sum(np.array(weights))/total)
    else:
        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                question = obj["question"]
                candidates = obj["candidates"]
                ground = obj["ground_truth"]
                filter_marks = filter_data(question, candidates, ground, mode, uncertainty_threshold, model, answer_is_num)
                if filter_marks is None:
                    filter_marks = [0] * len(candidates)
                weights.append(filter_marks)
    return weights

def load_samples(path = "/home/LAB/jiangcy/AdaDF/samples/gsm8k_test.jsonl"):
    q_list = []
    q_cnt = 0
    a_list = []
    a_cnt = 0
    a2q_index = {}
    jsonl_path = path
    with jsonlines.open(jsonl_path, "r") as reader:
        for obj in reader:
            question = obj["question"]
            candidates = obj["candidates"]
            ground = obj["ground_truth"]
            filtered_candidates = filter_data(question, candidates, ground)
            if filtered_candidates is None:
                continue
            # print(question)
            # for cand in filtered_candidates:
            #     print(cand)
            # quit()
            for cand in filtered_candidates:
                a2q_index[a_cnt] = q_cnt
                a_list.append(cand)
                a_cnt += 1
            q_list.append(question)
            q_cnt += 1
    return q_list, a_list, a2q_index

def dump_filtered_samples(path, q_list, a_list, a2q_index):
    file_path = path
    with jsonlines.open(file_path,"w") as writer:
        for i, a in enumerate(a_list):
            answer = a
            question = q_list[a2q_index[i]]
            d = {
                'context':'',
                'question': question,
                'answer': answer
            }
            writer.write(d)

# q_list, a_list, a2q_index = load_samples(path="/home/LAB/jiangcy/AdaDF/samples/gsm8k.jsonl")
# dump_filtered_samples("/home/LAB/jiangcy/AdaDF/filtered_samples/gsm8k_train_llama2-7b_filtered.jsonl", q_list, a_list, a2q_index)
# quit()

# for e in range(EPOCHS):
#     indices = list(range(sample_size))
#     random.shuffle(indices)
#     for i in range(0, sample_size, BATCH_SIZE):
#         if i + BATCH_SIZE >= sample_size:
#             end = sample_size
#         else:
#             end = i + BATCH_SIZE
#         batch_indices = indices[i: end]
#         batch_answers = [a_list[i] for i in batch_indices]
#         batch_questions = [q_list[a2q_index[i]] for i in batch_indices]
#         for i in range(len(batch_answers)):
#             print(batch_questions[i])
#             print(batch_answers[i])
#             print("--------")
#         quit()
#         print(batch_questions)
#         print(batch_answers)
#         quit()

