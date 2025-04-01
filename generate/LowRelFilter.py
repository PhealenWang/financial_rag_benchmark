import json
import os
from tqdm import tqdm
from argparse import ArgumentParser

'''
只保留相关分数高于lower_bound的文本
输入为score.json，输出为score_rel.json
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lower_bound', type=int, default=6)
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')

    args = parser.parse_args()
    lower_bound = args.lower_bound
    retriever = args.retriever

    file_dir = 'rel_docs'
    score_file = 'score.json'
    score_rel_file = 'score_rel.json'

    for root, dirs, files in tqdm(os.walk(file_dir)):
        # 计算当前目录的深度
        depth = root[len(file_dir):].count(os.sep)
        if depth == 3:  # 第三层子文件夹
            score_rel_path = os.path.join(root, retriever, score_rel_file)
            # 当score文件不存在时 进行操作，以防止重复
            if not os.path.exists(score_rel_path):
                # print(score_rel_path)
                with open(os.path.join(root, retriever, score_file), 'r', encoding='utf-8') as fr:
                    scores = json.load(fr)
                scores_rel = [[doc, score] for doc, score in scores if score >= 6]
                with open(score_rel_path, 'w', encoding='utf-8') as fw:
                    json.dump(scores_rel, fw, ensure_ascii=False, indent=4)

