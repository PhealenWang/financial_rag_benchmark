import json
import os
from tqdm import tqdm
from argparse import ArgumentParser
from openai import OpenAI

class RelevanceScorer(object):
    '''
    用LLM为每个查询对应的相关文档的相关性进行打分，分数为0-9的整数
    '''
    def __init__(self, model, prompt):
        self.model = model
        if self.model == '<- YOUR MODEL ->':
            self.client = OpenAI(api_key="<- YOUR API KEY ->", base_url="<- YOUR BASE URL ->")
        else:
            exit(1)
        with open(f'prompts/{prompt}', 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()

    def scorer(self, info):
        prompt = self.prompt.format(**info)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你精通金融领域的知识，能够快速确定一个文本和一个查询的相关性"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return int(response.choices[0].message.content.strip())


'''
为每个query对应的merge.json中的文档使用LLM打分
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='doubao1.5-pro-32k')
    parser.add_argument('--prompt', type=str, default='relevance_scorer')
    args = parser.parse_args()

    file_dir = 'rel_docs'
    merge_file = 'merge.json'
    score_file = 'score.json'

    scorer = RelevanceScorer(args.model, merge_file)

    for root, dirs, files in tqdm(os.walk(file_dir)):
        # 计算当前目录的深度
        depth = root[len(file_dir):].count(os.sep)
        if depth == 3:  # 第三层子文件夹
            score_path = os.path.join(root, score_file)
            # 当score文件不存在时 进行操作，以防止重复
            if not os.path.exists(score_path):
                query = os.path.basename(root)
                scores = {}
                with open(os.path.join(root, merge_file), 'r', encoding='utf-8') as fr:
                    docs = json.load(fr)
                for doc in tqdm(docs, desc=root):
                    try:
                        scores[doc] = scorer.scorer({'doc': doc, 'query': query})
                    except:
                        scores[doc] = -1
                    finally:
                        continue

                with open(os.path.join(root, score_file), 'w', encoding='utf-8') as fw:
                    # 对其进行逆序排序，相关性较高的在前
                    json.dump(sorted(scores.items(), key=lambda item: item[1], reverse=True), fw, ensure_ascii=False, indent=4)