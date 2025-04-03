import os
import json
from openai import OpenAI
from tqdm import tqdm
from argparse import ArgumentParser


class BestResponse(object):
    '''
    用LLM为每个查询生成回答
    '''
    def __init__(self, model):
        self.model = model
        if self.model == '<- YOUR MODEL ->':
            self.client = OpenAI(api_key="<- YOUR API KEY ->", base_url="<- YOUR BASE URL ->")
        else:
            exit(1)
        with open(f'prompts/best_response', 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()

    def response(self, info):
        prompt = self.prompt.format(**info)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0
        )
        return response.choices[0].message.content.strip()

'''
对文字内容进行评价，使用model based的指标
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    parser.add_argument('--model', type=str, default='deepseek-chat')
    args = parser.parse_args()

    # 所比较的LLMs
    models = ['moonshot-v1-8k', 'Baichuan4-Air', 'doubao1.5-pro-32k', 'deepseek-chat', 'deepseek-r1']
    n = len(models)

    rel_docs_folder = 'rel_docs'
    # 待判断的模型生成的结果
    result_path = os.path.join(rel_docs_folder, f'results/{args.retriever}/{{}}/content.jsonl')

    evaluation_folder = os.path.join(rel_docs_folder, f'results/{args.retriever}/groundtruth')
    os.makedirs(evaluation_folder, exist_ok=True)
    evaluation_path = os.path.join(evaluation_folder, 'content.jsonl')

    # 存放当前所有大模型的回答
    current_responses = [[] for _ in range(n)]
    for i in range(n):
        with open(result_path.format(models[i]), 'r', encoding='utf-8') as fr:
            current_responses[i] = fr.readlines()
    # query总数
    total = len(current_responses[0])

    # 如果文件已经存在，则读出
    if os.path.exists(evaluation_path):
        with open(evaluation_path, 'r', encoding='utf-8') as fr:
            evaluation_raw = fr.readlines()
    else:
        evaluation_raw = []

    judger = BestResponse(args.model)

    with open(evaluation_path, 'a', encoding='utf-8') as fw:
        # 断点继续
        for i in tqdm(range(len(evaluation_raw), total)):
            info = [json.loads(current_responses[j][i]) for j in range(n)]
            query = info[0]['query']
            first_intent = info[0]['first_intent']
            second_intent = info[0]['second_intent']
            docs = [t['answer'] for t in info]

            response = judger.response({'info': json.dumps({'query': query, 'answer': docs}, ensure_ascii=False)})
            row = {'query': query, 'model': None, 'answer': None, 'first_intent': first_intent, 'second_intent': second_intent, 'type': 'content'}
            res = json.loads(response)['label']
            if 0 <= res < n:
                row['model'] = models[res]
                row['answer'] = docs[res]
            fw.write(json.dumps(row, ensure_ascii=False) + '\n')


