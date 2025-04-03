import os
import json
from openai import OpenAI, RateLimitError
from zhipuai import ZhipuAI
from typing import Optional
from tqdm import tqdm
from time import sleep
from argparse import ArgumentParser


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
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    parser.add_argument('--metric', type=str, default='completeness', choices=['overall_pairwise', 'completeness'])
    parser.add_argument('--query_type', type=str, default='content', choices=['content'])
    args = parser.parse_args()

    model = '<- YOUR JUDGER MODEL ->'


    hypothesis_model = 'groundtruth'
    reference_model = '<- YOUR MODEL ->'

    rel_docs_folder = 'rel_docs/v7/'
    hypothesis_file = f'results/{args.retriever}/{hypothesis_model}/content.jsonl'
    reference_file = f'results/{args.retriever}/{reference_model}/content.jsonl'
    # 前为hypothesis，后为reference
    evaluation_folder = os.path.join(rel_docs_folder, f'evaluation/{args.retriever}/{hypothesis_model}_vs_{reference_model}/')
    evaluation_file = f'{args.metric}_details.jsonl'

    evaluator = PairwiseLLMEvaluator(metric=args.metric, model=model)

    with open(os.path.join(rel_docs_folder, hypothesis_file), 'r', encoding='utf-8') as fr:
        hypothesis_raw = fr.readlines()
    with open(os.path.join(rel_docs_folder, reference_file), 'r', encoding='utf-8') as fr:
        reference_raw = fr.readlines()

    if len(hypothesis_raw) != len(reference_raw):
        exit(1)

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
        for i in tqdm(range(len(evaluation_raw), len(hypothesis_raw))):
            row = json.loads(hypothesis_raw[i])
            hypothesis_row = json.loads(hypothesis_raw[i])
            reference_row = json.loads(reference_raw[i])

            if hypothesis_row['answer'] == '' or hypothesis_row['answer'] is None:
                row['evaluation'] = None
                score['invalid'] += 1
            else:
                row['evaluation'] = None
                # 文字内容，使用model-based指标进行评分
                if hypothesis_row['type'] == 'content':
                    try:
                        # 读取出相关文本
                        if args.retriever == 'base':
                            with open(os.path.join(rel_docs_folder, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/score_rel_6.json'), 'r') as fr:
                                docs_items = json.load(fr)
                            docs = [docs_items[i][0] for i in range(min(10, len(docs_items)))]
                        # bing
                        else:
                            with open(os.path.join(rel_docs_folder, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/merge.json'), 'r') as fr:
                                docs = json.load(fr)

                        if reference_row['answer'] == '' or reference_row['answer'] is None:
                            row['evaluation'] = [{args.metric: 0}]
                        else:
                            info = {'query': hypothesis_row['query'], 'hypothesis': hypothesis_row['answer'],
                                    'reference': reference_row['answer'], 'docs': docs}
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


