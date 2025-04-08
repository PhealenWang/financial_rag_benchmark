import os
import json
from openai import OpenAI, RateLimitError
from tqdm import tqdm
from time import sleep
from typing import Optional
from argparse import ArgumentParser


class PointwiseLLMEvaluator(object):
    '''
    detect hallucination
    '''
    def __init__(self, model, metric='overall'):
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

    def evaluate(self, info: dict) -> [Optional[float], dict]:
        '''
        对当前幻觉情况进行评分
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
                    if self.metric == 'overall':
                        # print(response.choices[0].message.content.strip())
                        overall_hallucination = json.loads(response.choices[0].message.content.strip())
                        return overall_hallucination['hallucination_check'], {}
                    if self.metric == 'relevance':
                        relevance = json.loads(response.choices[0].message.content.strip())
                        return relevance['relevance_check'], {}
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    parser.add_argument('--metric', type=str, default='relevance', choices=['overall', 'relevance'])
    parser.add_argument('--query_type', type=str, default='content', choices=['content'])
    args = parser.parse_args()

    # 判别LLM
    judger_model = '<- YOUR JUDGER MODEL ->'
    rel_dir = 'rel_docs/v7'
    model = '<- YOUR MODEL ->'

    # 这个地方需要读取出每个query是否是content类型，但deepseek-chat中加入了这个信息
    dataset_path = os.path.join(rel_dir, f'results/{args.retriever}/{model}/content.jsonl')
    evaluate_dir = os.path.join(rel_dir, f'evaluation/{args.retriever}/{model}')
    os.makedirs(evaluate_dir, exist_ok=True)
    evaluate_result_path = os.path.join(evaluate_dir, f'{args.metric}_details.jsonl')

    # 读取生成结果
    with open(dataset_path, 'r') as fr:
        dataset = fr.readlines()

    score = {args.metric: 0.0, 'count': 0, 'invalid': 0}
    # 如果文件已经存在，则读出。这里的逻辑是为计算总的平均分数，同时也考虑断点重传
    if os.path.exists(evaluate_result_path):
        with open(evaluate_result_path, 'r', encoding='utf-8') as fr:
            evaluation_raw = fr.readlines()
        # 读入已有的评价分数，计算总平均分
        for line in evaluation_raw:
            info = json.loads(line)
            if info['evaluation'] is not None and info['evaluation'][0][args.metric] != -1:
                score[args.metric] += info['evaluation'][0][args.metric]
                score['count'] += 1
            else:
                score['invalid'] += 1
    else:
        evaluation_raw = []

    evaluator = PointwiseLLMEvaluator(model=judger_model, metric=args.metric)

    with open(evaluate_result_path, 'a', encoding='utf-8') as fw:
        # 断点继续
        for i in tqdm(range(len(evaluation_raw), len(dataset))):
            row = json.loads(dataset[i])
            dataset_row = json.loads(dataset[i])

            if dataset_row['answer'] == '' or dataset_row['answer'] is None:
                row['evaluation'] = None
                score['invalid'] += 1
            else:
                row['evaluation'] = None
                # 文字内容，使用model-based指标进行评分
                # print(dataset_row)
                if dataset_row['type'] == 'content':
                    try:
                        # 读取出相关文本
                        if args.retriever == 'base':
                            with open(os.path.join(rel_dir, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/score_rel_6.json'), 'r') as fr:
                                docs_items = json.load(fr)
                            docs = [docs_items[i][0] for i in range(min(10, len(docs_items)))]
                        else:
                            with open(os.path.join(rel_dir, f'{row['first_intent']}/{row['second_intent']}/{row['query']}/{args.retriever}/merge.json'), 'r') as fr:
                                docs = json.load(fr)

                        info = {'query': dataset_row['query'], 'docs': docs, 'answer': dataset_row['answer']}
                        current_score, other = evaluator.evaluate(info)
                        score['count'] += 1
                        row['evaluation'] = [{args.metric: current_score, 'other': other}]
                        score[args.metric] += row['evaluation'][0][args.metric]
                    except:
                        score['invalid'] += 1
                    finally:
                        del row['answer']
                        fw.write(json.dumps(row, ensure_ascii=False) + '\n')

        score[args.metric] /= score['count']
        fw.write(json.dumps(score, ensure_ascii=False) + '\n')

