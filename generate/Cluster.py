import json
import os
from argparse import ArgumentParser
from tqdm import tqdm


'''
把各个query及其回答汇总到一个文件当中
这里有一个小bug，在遍历文件的时候，也会遍历evaluation和results文件夹，但因为这两个文件夹下并没有需要的文件，所以会被跳过
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-chat')
    parser.add_argument('--close', type=bool, default=False, help='是否提供相关文本')
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    args = parser.parse_args()


    file_dir = 'rel_docs'
    # 大模型生成的结果对应的文件
    result_file = f'result_{args.model}.json' if not args.close else f'result_{args.model}_close.json'
    result_dir = os.path.join(file_dir, f'results/{args.retriever}/{args.model}' if not args.close else f'results/{args.retriever}/{args.model}_close')
    query_base_file = 'query_base/query_base.json'

    os.makedirs(result_dir, exist_ok=True)
    # 将content型和value型分开存放
    content_path = os.path.join(result_dir, 'content.jsonl')
    value_path = os.path.join(result_dir, 'value.jsonl')

    with open(query_base_file, 'r', encoding='utf-8') as fr:
        query_base = json.load(fr)


    with open(content_path, 'w', encoding='utf-8') as fw_content, open(value_path, 'w', encoding='utf-8') as fw_value:
        for root, dirs, files in tqdm(os.walk(file_dir)):
            # 计算当前目录的深度
            depth = root[len(file_dir):].count(os.sep)
            if depth == 3:  # 第三层子文件夹
                # 拆分文件名 最后三个元素以此为：一级意图 二级意图 query
                temp = os.path.normpath(root).split(os.sep)
                if temp[-3] in ['results', 'evaluation']:
                    continue
                result_path = os.path.join(root, args.retriever, result_file)

                text = ''
                try:
                    if os.path.exists(result_path):
                        with open(result_path, 'r', encoding='utf-8') as fr:
                            result = json.load(fr)
                        text = result[temp[-1]]
                except json.decoder.JSONDecodeError:
                    print(result_path)
                finally:

                    to_write = {'query': temp[-1], 'answer': text, 'first_intent': temp[-3], 'second_intent': temp[-2]}

                    # 这段逻辑是需要query base的相关内容来判断当前的query是否使用api
                    # 如果使用api，则type字段为value，没有使用api，type字段为content
                    Exit = False
                    # print(temp[-3], query_base[temp[-3]])
                    for _, query_info in query_base[temp[-3]]['二级意图'][temp[-2]].items():
                        if isinstance(query_info, dict) and temp[-1] in query_info.keys() and 'api' in query_info[temp[-1]].keys():
                            to_write['type'] = 'value'
                            fw_value.write(json.dumps(to_write, ensure_ascii=False) + '\n')
                            Exit = True
                            break
                    if Exit == False:
                        to_write['type'] = 'content'
                        fw_content.write(json.dumps(to_write, ensure_ascii=False) + '\n')

