import os
import json
from openai import OpenAI, RateLimitError
from typing import Optional
from tqdm import tqdm
from time import sleep
from argparse import ArgumentParser
import pandas as pd


class PairwiseLLMEvaluator(object):
    '''
    model based evaluation method
    '''
    def __init__(self, model, metric):
        '''
        :param model: 所使用的LLM
        :param metric: 所使用的指标
        '''
        self.model = model
        if self.model == '<- YOUR MODEL ->':
            self.client = OpenAI(api_key="<- YOUR API KEY ->", base_url="<- YOUR BASE URL ->")
        else:
            exit(1)

        self.metric = metric
        # 读入prompt
        prompt_path = f'prompts/{metric}'
        if not os.path.exists(prompt_path):
            exit(101)
        with open(prompt_path, 'r') as fr:
            self.prompt = fr.read()

    def score(self, info: dict) -> [Optional[float], dict]:
        '''
        对对应的指标进行评分
        :param info
        :return:
        '''

        prompt = self.prompt.format(**info)

        count = 1
        while count < 3:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # max_tokens=2048,
                    temperature=0
                )
                try:
                    if self.metric == 'overall_pairwise':
                        # print(response.choices[0].message.content.strip())
                        overall_hallucination = json.loads(response.choices[0].message.content.strip())
                        return overall_hallucination['hallucination_check'], {}
                    elif self.metric == 'completeness':
                        completeness = json.loads(response.choices[0].message.content.strip())
                        return completeness['completeness_check'], {}
                    else:
                        return -1, {}
                except:
                    return None, {'content': response.choices[0].message.content.strip()}
            except RateLimitError:
                count += 1
                print('429...sleeping')
                sleep(20)
                continue
        print('too many requests')
        exit(1)


'''
对文字内容进行评价，使用model based的指标
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing', 'bert'], help='retriever type')
    parser.add_argument('--metric', type=str, default='completeness', choices=['overall_pairwise', 'completeness'])
    parser.add_argument('--query_type', type=str, default='content', choices=['content'])
    args = parser.parse_args()

    model = '<- YOUR JUDGER MODEL ->'


    hypothesis_model = 'groundtruth'
    reference_model = '<- YOUR MODEL ->'

    rel_docs_folder = 'rel_docs'
    hypothesis_file = f'dataset/content.jsonl'
    reference_file = f'{rel_docs_folder}/results/{args.retriever}/{reference_model}/content.jsonl'

    # 读入文件
    # df1为groundtruth（主表），df2为当前的result（次表）
    df1 = pd.read_json(hypothesis_file, lines=True)
    df2 = pd.read_json(reference_file, lines=True)

    # 可只保留需要的字段，比如 query 和 answer
    df2 = df2.drop_duplicates(subset='query', keep='first')
    df2 = df2[['query', 'answer']].rename(columns={'answer': 'answer_current'})

    # 按query左连接
    merged = pd.merge(df1, df2, on='query', how='left')

    # 如果你想把 NaN 替换成空字符串
    merged = merged.fillna('')

    records = merged.to_dict(orient="records")

    # 前为hypothesis，后为reference
    evaluation_folder = os.path.join(rel_docs_folder, f'evaluation/{args.retriever}/{hypothesis_model}_vs_{reference_model}/')
    evaluation_file = f'{args.metric}_details.jsonl'

    evaluator = PairwiseLLMEvaluator(metric=args.metric, model=model)

    score = {args.metric: 0, "count": 0, 'invalid': 0}

    os.makedirs(evaluation_folder, exist_ok=True)
    # 如果文件已经存在，则读出。这里的逻辑是为计算总的平均分数，同时也考虑断点重传
    if os.path.exists(os.path.join(evaluation_folder, evaluation_file)):
        with open(os.path.join(evaluation_folder, evaluation_file), 'r', encoding='utf-8') as fr:
            evaluation_raw = fr.readlines()
        # 读入已有的评价分数，计算总平均分
        for line in evaluation_raw:
            info = json.loads(line)
            if info['evaluation'] is not None:
                score[args.metric] += info['evaluation'][0][args.metric]
                score['count'] += 1
            else:
                score['invalid'] += 1
    else:
        evaluation_raw = []

    count = 0
    with open(os.path.join(evaluation_folder, evaluation_file), 'a', encoding='utf-8') as fw:
        # 断点继续
        for i in tqdm(range(len(evaluation_raw), len(records))):
            row = records[i]

            if row['answer'] == '' or row['answer'] is None:
                row['evaluation'] = None
                score['invalid'] += 1
            else:
                row['evaluation'] = None
                # 文字内容，使用model-based指标进行评分
                if row['type'] == 'content':
                    try:
                        # 读取出相关文本
                        if args.retriever == 'base':
                            with open(os.path.join(rel_docs_folder, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/score_rel_6.json'), 'r') as fr:
                                docs_items = json.load(fr)
                            docs = [docs_items[i][0] for i in range(min(10, len(docs_items)))]
                        # bing
                        elif args.retriever == 'bing':
                            with open(os.path.join(rel_docs_folder, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/merge.json'), 'r') as fr:
                                docs = json.load(fr)
                        # bert
                        else:
                            with open(os.path.join(rel_docs_folder, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/origin_docs.json'), 'r') as fr:
                                docs = json.load(fr)

                        if row['answer_current'] == '' or row['answer_current'] is None:
                            row['evaluation'] = [{args.metric: 0}]
                        else:
                            info = {'query': row['query'], 'hypothesis': row['answer'],
                                    'reference': row['answer_current'], 'docs': docs}
                            current_score, other = evaluator.score(info)
                            row['evaluation'] = [{args.metric: current_score, 'other': other}]
                            score[args.metric] += row['evaluation'][0][args.metric]
                        score['count'] += 1
                    except:
                        score['invalid'] += 1
                    finally:
                        del row['answer']
                        fw.write(json.dumps(row, ensure_ascii=False) + '\n')

        score[args.metric] /= score['count']
        fw.write(json.dumps(score, ensure_ascii=False) + '\n')


