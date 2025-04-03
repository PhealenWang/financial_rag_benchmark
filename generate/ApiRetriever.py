from generate.ApiTushare import ApiTushare
import json
import os
from argparse import ArgumentParser

'''
Api检索器，其主要代码在utils/ApiTushare.py中
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='文本检索器的类型')
    args = parser.parse_args()
    retriever = args.retriever

    query_base_file = 'query_base/query_base.json'
    rel_docs_folder = 'rel_docs'

    with open(query_base_file, 'r') as f:
        query_base = json.load(f)

    obj = ApiTushare()

    for first_intent, second in query_base.items():
        first_dir = os.path.join(rel_docs_folder, first_intent)
        os.makedirs(first_dir, exist_ok=True)
        for second_intent, base_queries in second['二级意图'].items():
            second_dir = os.path.join(first_dir, second_intent)
            os.makedirs(second_dir, exist_ok=True)
            for base, queries in base_queries.items():
                if base == 'count':
                    break
                for query, params in queries.items():
                    query_dir = os.path.join(second_dir, query, retriever)
                    os.makedirs(query_dir, exist_ok=True)

                    # 处理API数据
                    api_data = os.path.join(query_dir, f'dataframe.csv')
                    column_name_explanation = os.path.join(query_dir, f'column.json')
                    # 如果当前已经存在 则跳过
                    if os.path.exists(api_data) and os.path.exists(column_name_explanation):
                        continue
                    if params.get('api') is None or params['api']['use'] == False:
                        continue

                    # 找到需要调用的API函数
                    method = getattr(obj, params['api']['api'])
                    df, columns = method(**params['api']['params'])
                    df.to_csv(api_data, index=False)
                    with open(column_name_explanation, 'w') as fw:
                        json.dump(columns, fw, indent=4, ensure_ascii=False)
