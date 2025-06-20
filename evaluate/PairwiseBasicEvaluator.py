from rouge import Rouge
import os
import json
import re
from argparse import ArgumentParser
import sacrebleu
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class PairwiseBasicEvaluator(object):
    def __init__(self, metric, query_type):
        self.metric = metric
        self.query_type = query_type
        if self.metric == 'cos-sim' and self.query_type == 'content':
            self.vectorizer = TfidfVectorizer()
        if self.metric == 'rouge-l' and self.query_type == 'content':
            self.rouger = Rouge()
        elif self.metric == 'bleu' and self.query_type == 'content':
            pass
        elif self.metric == 'accuracy' and self.query_type == 'value':
            pass

    def score(self, hypothesis, reference):
        '''
        给出当前评价结果的值
        :param hypothesis:
        :param reference:
        :return:
        '''
        if self.metric == 'cos-sim' and self.query_type == 'content':
            # 将分词后的文本转换为字符串，以符合TfidfVectorizer的输入要求
            tokenized_standard = " ".join(list(jieba.cut(hypothesis)))
            tokenized_generated = " ".join(list(jieba.cut(reference)))
            # 使用TF-IDF向量化文本
            tfidf_matrix = self.vectorizer.fit_transform([tokenized_standard, tokenized_generated])
            # 计算余弦相似度
            similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return similarity_score
        elif self.metric == 'rouge-l' and self.query_type == 'content':
            scores = self.rouger.get_scores(hypothesis, reference)
            return scores[0]['rouge-l']['f']
        elif self.metric == 'bleu' and self.query_type == 'content':
            reference_seg = [" ".join(jieba.lcut(text)) for text in hypothesis]
            candidate_seg = " ".join(jieba.lcut(reference))
            # 计算 BLEU 分数
            bleu_score = sacrebleu.corpus_bleu([candidate_seg], [reference_seg])
            # 计算BLEU评分
            return bleu_score.score
        elif self.metric == 'accuracy' and self.query_type == 'value':
            # 定义正则表达式模式，用于匹配数字和单位
            pattern = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z\u4e00-\u9fa5]+)?'
            # 使用正则表达式匹配字符串中的数字和单位
            match1 = re.match(pattern, hypothesis)
            match2 = re.match(pattern, reference)

            # 第一类：检查是否两个字符串都成功匹配到数字
            if match1 and match2:
                # 提取数字并转换为浮点数
                num1 = float(match1.group(1))
                num2 = float(match2.group(1))
                # 提取单位
                unit1 = match1.group(2)
                unit2 = match2.group(2)

                # 二者数值相同
                if num1 == num2 or (num1 != 0 and abs((num1 - num2) / num1) < 1e-5):
                    # 三种情况：1. 二者数字相同且都没有单位；2. 二者数字相同且其中一个没有单位；3. 二者数字相同且单位相同
                    if (unit1 is None and unit2 is None) or (unit1 is None or unit2 is None) or (unit1 == unit2):
                        return 1
                    else:
                        return 0

            # 第二类：没有匹配到前面的数字
            # 情况4: str1是str2的前缀
            if len(hypothesis) <= len(reference) and reference.startswith(hypothesis):
                return 1
            # 情况5: 二者没有数字且“单位”相同
            str1 = hypothesis.replace(' ', '')
            str2 = reference.replace(' ', '')
            if ',' in str1:
                set1 = set(str1.split(','))
            elif '、' in str1:
                set1 = set(str1.split('、'))
            else:
                return 0
            if ',' in str2:
                set2 = set(str2.split(','))
            elif '、' in str2:
                set2 = set(str2.split('、'))
            else:
                return 0
            if set1 == set2:
                return 1
            # 如果不满足上述情况，则返回 False
            return 0
        else:
            return -1


'''
对生成的答案进行评价 使用rouge对文字内容进行评价，使用accuracy对数值内容进行评价
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing', 'bert'], help='retriever type')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['rouge-l', 'accuracy', 'bleu', 'cos-sim'])
    parser.add_argument('--query_type', type=str, default='value', choices=['content', 'value'])
    args = parser.parse_args()

    hypothesis_model = 'groundtruth'
    reference_model = '<- YOUR MODEL ->'

    rel_docs_folder = 'rel_docs'
    hypothesis_dir = 'dataset'
    reference_dir = f'results/{args.retriever}/{reference_model}'
    # 前为hypothesis，后为reference
    evaluation_folder = os.path.join(rel_docs_folder, f'evaluation/{args.retriever}/{hypothesis_model}_vs_{reference_model}/')
    os.makedirs(evaluation_folder, exist_ok=True)

    evaluator = PairwiseBasicEvaluator(args.metric, args.query_type)
    
    read_file = f'{args.query_type}.jsonl'

    # 读入文件
    # df1为groundtruth（主表），df2为当前的result（次表）
    df1 = pd.read_json(os.path.join(hypothesis_dir, read_file), lines=True)
    df2 = pd.read_json(os.path.join(rel_docs_folder, reference_dir, read_file), lines=True)

    # 可只保留需要的字段，比如 query 和 answer
    df2 = df2.drop_duplicates(subset='query', keep='first')
    df2 = df2[['query', 'answer']].rename(columns={'answer': 'answer_current'})

    # 按query左连接
    merged = pd.merge(df1, df2, on='query', how='left')

    # 如果你想把 NaN 替换成空字符串
    merged = merged.fillna('')

    records = merged.to_dict(orient="records")
    # 输出文件
    evaluation_file = f'{args.metric}_details.jsonl'

    score = {args.metric: 0, "count": 0, 'invalid': 0}
    count = 0

    with open(os.path.join(evaluation_folder, evaluation_file), 'w', encoding='utf-8') as fw:
        for record in records:
            row = record
            row['evaluation'] = None
            if record['answer'] is None or record['answer'] == '':
                score['invalid'] += 1
            else:
                # 统一评分
                if record['type'] == args.query_type:
                    score['count'] += 1
                    if record['answer_current'] == '':
                        row['evaluation'] = [{args.metric: 0}]
                    else:
                        temp_score = evaluator.score(record['answer'], record['answer_current'])
                        # print(temp_score)
                        if temp_score == -1:
                            row['evaluation'] = [{args.metric: 0}]
                        else:
                            row['evaluation'] = [{args.metric: temp_score, 'groundtruth': record['answer'], 'current': record['answer_current']}]
                            score[args.metric] += row['evaluation'][0][args.metric]

            del row['answer']
            del row['answer_current']
            fw.write(json.dumps(row, ensure_ascii=False) + '\n')

        if score['count'] > 0:
            score[args.metric] /= score['count']
        fw.write(json.dumps(score, ensure_ascii=False) + '\n')


