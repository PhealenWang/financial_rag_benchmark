import json
from argparse import ArgumentParser
import pandas as pd
from openai import OpenAI


class QueryIntentClassifier(object):
    '''
    用LLM标注已有query的意图
    '''
    def __init__(self, model, prompt):
        self.model = model
        if self.model == 'doubao1.5-pro-32k':
            self.client = OpenAI(api_key="951def3b-e049-43cf-8f83-bcac8457eb78", base_url="https://ark.cn-beijing.volces.com/api/v3")
        else:
            exit(1)
        with open(f'prompts/{prompt}', 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()

    def classify(self, info):
        prompt = self.prompt.format(**info)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个非常敏锐的投资顾问，你擅长将用户的问题进行意图分类"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content.strip()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='doubao1.5-pro-32k')
    parser.add_argument('--prompt', type=str, default='intent_classify')
    parser.add_argument('--input_data', type=str, help='待意图分类的查询')
    parser.add_argument('--output_data', type=str, help='分类好的查询对应的文件')
    args = parser.parse_args()

    classifier = QueryIntentClassifier(args.model, args.prompt)

    # 所有存在的意图
    with open('query_base/intents.json', 'r', encoding='utf-8') as fr:
        intent_categories = json.load(fr)

    # 读入包含query的文件
    df = pd.read_csv(args.input_data)
    # 输出的文件
    df_output = []

    for _, row in df.iterrows():
        intents = classifier.classify({'intent_categories': intent_categories, 'query': row['query']})
        df_output.append({'query': row['query'], 'intent': intents})

    df_output = pd.DataFrame(df_output)
    df_output.to_csv(args.output_data, index=False)

