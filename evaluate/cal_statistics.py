import os
import json
from collections import defaultdict
from copy import deepcopy

'''
对已有的评价结果进行数据统计，按照意图来进行分别统计
'''
if __name__ == '__main__':
    metrics = ['rouge-l', 'accuracy']
    hypothesis_model = 'deepseek-chat'
    reference_model = 'deepseek-chat_close'
    # reference_model = 'moonshot-v1-8k'
    # reference_model = 'Baichuan4-Air'

    query_base_file = 'query_base/query_base.json'
    evaluation_folder = f'rel_docs/v6/evaluation/{hypothesis_model}_vs_{reference_model}/'
    evaluation_file = f'{hypothesis_model}_vs_{reference_model}.jsonl'


    # 将query base文件处理为字典形式：一级意图 - 二级意图 - {指标值，计数}
    with open(query_base_file, 'r', encoding='utf-8') as query_base:
        origin_base = json.load(query_base)
    statistics = defaultdict(dict)
    for first_intent, second in origin_base.items():
        statistics[first_intent] = defaultdict(dict)
        for second_intent, _ in second['二级意图'].items():
            statistics[first_intent][second_intent] = {'value': 0, 'count': 0}

    # 读出所需要统计的文件的内容，去除最后一行
    with open(os.path.join(evaluation_folder, evaluation_file), 'r', encoding='utf-8') as fr:
        evaluations = fr.readlines()
    # 最后一行是总的统计结果
    n = len(evaluations) - 1

    for metric in metrics:
        metric_file = f'{metric}.json'
        current_statistics = deepcopy(statistics)

        for i in range(n):
            info = json.loads(evaluations[i])
            if info['evaluation'] is not None and metric in info['evaluation'][0].keys():
                if metric == 'rouge-l':
                    current_statistics[info['first_intent']][info['second_intent']]['value'] = info['evaluation'][0][metric]['f']
                else:
                    current_statistics[info['first_intent']][info['second_intent']]['value'] = info['evaluation'][0][
                        metric]
                current_statistics[info['first_intent']][info['second_intent']]['count'] += 1
        # print(current_statistics)
        for first_intent, second in current_statistics.items():
            for second_intent, _ in second.items():
                current_statistics[first_intent][second_intent]['value'] = second[second_intent]['value'] / second[second_intent]['count'] if second[second_intent]['count'] > 0 else 0
        with open(os.path.join(evaluation_folder, metric_file), 'w', encoding='utf-8') as fw:
            json.dump(current_statistics, fw, ensure_ascii=False, indent=4)

