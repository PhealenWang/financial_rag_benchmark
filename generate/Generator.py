import json
import os
from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
from openai import OpenAI, RateLimitError
from time import sleep


class Generator(object):
    '''
    用LLM为每个查询生成回答
    '''
    def __init__(self, model, prompt):
        self.model = model
        if self.model == 'doubao1.5-pro-32k':
            self.client = OpenAI(api_key="951def3b-e049-43cf-8f83-bcac8457eb78", base_url="https://ark.cn-beijing.volces.com/api/v3")
        else:
            exit(1)
        with open(f'prompts/{prompt}', 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()
        self.system = {'generator_close': '你是一个金融投资领域专业的助手，擅长回答金融投资方面各种问题，并且做出准确且简洁的回答',
                       'generator_value': '你精通金融领域的知识与相关数据分析，能够快速在pandas数据格式中找到问题的答案',
                       'generator_text': '你是一个金融投资领域专业的助手，擅长回答金融投资方面各种问题，并且会依据提供的相关文本信息做出准确且简洁的回答。'}

    def generate(self, info):
        prompt = self.prompt.format(**info)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system[self.prompt]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content.strip()

'''
为每个query生成回答
python Generator.py --model=deepseek-chat --retriever=bing --value=False
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-chat')
    parser.add_argument('--close', type=bool, default=False, help='是否提供相关文本')
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    parser.add_argument('--value', type=bool, default=True, choices=[True, False], help='是否计算value型query')
    args = parser.parse_args()


    file_dir = 'rel_docs'
    rel_texts_file = 'score_rel.json' if args.retriever == 'base' else 'merge.json'
    rel_dataframe_file = f'{args.retriever}/dataframe.csv'
    columns_info_file = f'{args.retriever}/column.json'
    result_file = f'result_{args.model}.json' if not args.close else f'result_{args.model}_close.json'

    # 根据三类 不同的任务，选择prompt
    if args.close:
        prompt = 'generator_close'
    else:
        if args.value:
            prompt = 'generator_value'
        else:
            prompt = 'generator_text'


    generator = Generator(args.model, prompt)

    for root, dirs, files in tqdm(os.walk(file_dir)):
        # 计算当前目录的深度
        depth = root[len(file_dir):].count(os.sep)
        if depth == 3:  # 第三层子文件夹
            query = os.path.basename(root)
            df_path = os.path.join(root, rel_dataframe_file)
            columns_info_path = os.path.join(root, columns_info_file)
            result_path = os.path.join(root, args.retriever, result_file)

            # 如果已经完成了结果的生成 就跳过本条query 相当于断点标记
            if os.path.exists(result_path):
                continue
            try:
                # 需要参考文本，即RAG
                if not args.close:
                    # 如果有api数据，则使用api进行生成
                    if os.path.exists(df_path) and os.path.exists(columns_info_path):
                        if not args.value:
                            continue
                        dataframe = pd.read_csv(os.path.join(root, rel_dataframe_file))
                        with open(columns_info_path, 'r') as fr:
                            columns_info = json.load(fr)
                        info = {'dataframe': dataframe, 'columns': json.dumps(columns_info, indent=4, ensure_ascii=False), 'query': query}
                    # 没有api数据，使用文本进行生成
                    else:
                        # continue
                        with open(os.path.join(root, args.retriever, rel_texts_file), 'r', encoding='utf-8') as fr:
                            docs_items = json.load(fr)
                        # 没有相关文本 跳过本条query
                        if len(docs_items) == 0:
                            continue
                        # 最多十个相关文档
                        docs = [docs_items[i][0] for i in range(min(10, len(docs_items)))] if args.retriever == 'base' else docs_items
                        info = {'docs': json.dumps(docs, indent=4, ensure_ascii=False), 'query': query}
                # 无需参考文本，即baseline
                else:
                    info = {'query': query}
                with open(result_path, 'w', encoding='utf-8') as fw:
                    json.dump({query: generator.generate(info)}, fw, ensure_ascii=False, indent=4)
            # 当前正在处理的这个query会被跳过，需要在最后再处理一次，专门处理429的问题
            except RateLimitError:
                sleep(5)
            finally:
                continue