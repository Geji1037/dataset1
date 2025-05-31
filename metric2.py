from typing import Dict, List
from rouge_score import rouge_scorer
from collections import defaultdict
import tqdm
import json
import os

def read_list(file, k):
    dic = {}
    lines = open(file).readlines()
    for line in lines:
        line = line.strip()
        data = json.loads(line)
        if data['category'] not in dic:
            dic[data['category']] = []
        tmpd = data[k]
        if tmpd.endswith('</s>'):
            tmpd = tmpd.split('</s>')[0]
        dic[data['category']].append(tmpd)
    return dic

# 替代 t5.metrics.rouge 的函数
def compute_rouge(targets: List[str], predictions: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = defaultdict(list)

    for target, pred in zip(targets, predictions):
        score = scorer.score(target, pred)
        for k in score:
            scores[k].append(score[k].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}

# 多参考情况下，挑选最佳 rouge
def rouge_fn(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    max_references = {rouge_type: [] for rouge_type in rouge_types}
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    for targ_for_resp, resp in tqdm.tqdm(zip(targets, predictions), total=len(targets)):
        resp_scores = [scorer.score(t, resp) for t in targ_for_resp]
        for rouge_type in rouge_types:
            best_score_index = max(range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type].fmeasure)
            best_ref = targ_for_resp[best_score_index]
            max_references[rouge_type].append(best_ref)

    results = {}
    for rouge_type in rouge_types:
        results[rouge_type] = compute_rouge(max_references[rouge_type], predictions)[rouge_type]
    return results

def rouge(targets, predictions):
    return compute_rouge(targets, predictions)

def get_result(targets, predictions, save):
    results = {}
    total_target = []
    total_pre = []
    for k in targets.keys():
        result = rouge(targets[k], predictions[k])
        results[k] = result
        total_target.extend(targets[k])
        total_pre.extend(predictions[k])
    results['total'] = rouge(total_target, total_pre)
    print(results)
    with open(save, 'w') as f:
        f.write(json.dumps(results, indent=2))

# 调用部分
targets = read_list('./data/dataset1/flan_test_200_selected_nstrict_1.jsonl', 'output')
predictions = read_list('./out/result2.jsonl', 'answer')
get_result(targets, predictions, './out/result_local.json')
