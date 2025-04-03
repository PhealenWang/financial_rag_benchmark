import os
import json
from argparse import ArgumentParser
import jieba
import kenlm


class PointwiseBasicEvaluator(object):
    def __init__(self, metric, query_type):
        self.metric = metric
        self.query_type = query_type
        if self.metric == 'perplexity' and self.query_type == 'content':
            self.model = kenlm.Model('models/zh.arpa.bin')

    def score(self, text):
        '''
        给出当前评价结果的值
        :param text:
        :return:
        '''
        if self.metric == 'perplexity' and self.query_type == 'content':
            words = list(jieba.cut(text))
            joined_text = " ".join(words)
            # 计算得分
            score = self.model.score(joined_text, bos=True, eos=True)
            # 计算词数
            word_count = len(words)
            if word_count == 0:
                return -1
            # 计算困惑度
            ppl = 10 ** (-score / word_count)
            return ppl
        else:
            return -1


'''

'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    parser.add_argument('--metric', type=str, default='perplexity', choices=['perplexity'])
    parser.add_argument('--query_type', type=str, default='content', choices=['content', 'value'])
    args = parser.parse_args()

    current_model = 'groundtruth'

    rel_docs_folder = 'rel_docs/v7/'
    current_dir = f'results/{args.retriever}/{current_model}'
    evaluation_folder = os.path.join(rel_docs_folder, f'evaluation/{args.retriever}/{current_model}/')
    os.makedirs(evaluation_folder, exist_ok=True)

    evaluator = PointwiseBasicEvaluator(args.metric, args.query_type)

    read_file = f'{args.query_type}.jsonl'

    with open(os.path.join(rel_docs_folder, current_dir, read_file), 'r', encoding='utf-8') as fr:
        current_raw = fr.readlines()

    # 输出文件
    evaluation_file = f'{args.metric}_details.jsonl'

    score = {args.metric: 0, "count": 0, 'invalid': 0}
    count = 0

    with open(os.path.join(evaluation_folder, evaluation_file), 'w', encoding='utf-8') as fw:
        for i in range(len(current_raw)):
            row = json.loads(current_raw[i])
            current_row = json.loads(current_raw[i])

            row['evaluation'] = None
            if current_row['answer'] is None or current_row['answer'] == '':
                score['invalid'] += 1
            else:
                # 统一评分
                if current_row['type'] == args.query_type:
                    # print(hypothesis_row['answer'], reference_row['answer'])
                    score['count'] += 1
                    temp_score = evaluator.score(current_row['answer'])
                    # print(temp_score)
                    if temp_score == -1:
                        score['invalid'] += 1
                        continue
                    else:
                        row['evaluation'] = [{args.metric: temp_score}]
                        score[args.metric] += row['evaluation'][0][args.metric]

            del row['answer']
            fw.write(json.dumps(row, ensure_ascii=False) + '\n')

        if score['count'] > 0:
            score[args.metric] /= score['count']
        fw.write(json.dumps(score, ensure_ascii=False) + '\n')


