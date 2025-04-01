import subprocess
import json
import os
from argparse import ArgumentParser
import time
from tqdm import tqdm
import requests
from datetime import datetime


class TextRetriever(object):
    '''
    文本类检索器
    '''
    def __init__(self, retriever: str):
        '''
        :param retriever: 检索器有两类，基本检索器（数据库查询），外部检索器（bing）
        '''
        self.type = retriever

    def _freshness(self, start_date: str, end_date: str) -> str:
        '''
        为bing的freshness参数计算起始时间
        :param start_date:
        :param end_date:
        :return:
        '''
        if start_date != '' and end_date != '':
            start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            return f"{start}..{end}"
        elif start_date == '' and end_date != '':
            end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            return f"..{end}"
        elif start_date != '' and end_date == '':
            start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
            return f"{start}.."
        else:
            return ""

    def request(self, query, params, count=10, offset=0, timeout=2000, limit=3):
        if self.type == 'bing':
            url = 'https://api.coze.cn/v1/workflow/run'
            key = 'pat_PTzVCHas1WmB8FQ0ICWpe2bCR0v73Fc75BUtBTYT5tMkgZBFhGef9PRoFxmia0BF'
            start_time = time.time()
            headers = {
                'Authorization': 'Bearer ' + key,
                'Content-Type': 'application/json'
            }

            jsonBody = {
                "parameters": {
                    "count": count,
                    "freshness": self._freshness(params["startdate"], params["enddate"]),
                    "offset": offset,
                    "query": query
                },
                "workflow_id": '7452660385951268874'
            }
            data = json.dumps(jsonBody, default=lambda obj: obj.__dict__, ensure_ascii=False)

            retry = 0
            while retry < limit:
                r = requests.post(url, data.encode('utf-8'), headers=headers, timeout=timeout)
                if r.status_code == requests.codes.ok:
                    print('BingClient cost = ' + str(time.time() - start_time))
                    return r.json()
                else:
                    print('[BingClient][ERROR] code is ' + str(r.status_code))
                    print('[BingClient] retry = ' + str(retry))
                    time.sleep(10)
                    retry += 1
                    continue

            print('BingClient cost = ' + str(time.time() - start_time))
            return None

        else:
            recallnum = 150
            size = 60
            command = [
                "curl",
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d",
                f'{{"query":["{query}"], "type":"report,wechat,news", "recallnum": {recallnum}, "size": {size}, "source": "wf", "startdate": "{params["startdate"]}", "enddate": "{params["enddate"]}"}}',
                "http://101.126.90.188:5629/embedding/search/v2"
            ]
            # 执行curl命令并捕获输出
            result = subprocess.run(command, text=True, capture_output=True)

            # 检查命令是否成功执行
            if result.returncode == 0:
                # 解析返回的内容
                response_content = result.stdout
                return response_content
            else:
                # 如果命令执行失败，打印错误信息
                print(f"Error: {result.stderr}")
                return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retriever', type=str, default='base', choices=['base', 'bing'], help='retriever type')
    args = parser.parse_args()
    retriever = args.retriever

    client = TextRetriever(retriever)

    query_base_file = 'query_base/query_base.json'
    rel_docs_folder = 'rel_docs'

    with open(query_base_file, 'r') as f:
        query_base = json.load(f)

    for first_intent, second in tqdm(query_base.items(), desc='first intent'):
        first_dir = os.path.join(rel_docs_folder, first_intent)
        os.makedirs(first_dir, exist_ok=True)
        for second_intent, base_queries in tqdm(second['二级意图'].items(), desc=f'first intent: {first_intent}'):
            second_dir = os.path.join(first_dir, second_intent)
            os.makedirs(second_dir, exist_ok=True)
            for base, queries in base_queries.items():
                if base == 'count':
                    break
                for query, params in queries.items():
                    # value型不作处理
                    if 'api' in params.keys():
                        continue
                    # 最新的query信息文件夹，由于与检索器相关，因此增加了检索器文件夹
                    query_dir = os.path.join(second_dir, query, retriever)
                    os.makedirs(query_dir, exist_ok=True)

                    json_path = os.path.join(query_dir, f'origin_docs.json')
                    # 如果当前json已经存在 则跳过
                    if os.path.exists(json_path):
                        continue

                    response_content = client.request(query, params)

                    if response_content:
                        # 解析返回的JSON内容
                        try:
                            with open(json_path, 'w') as fw:
                                json.dump(response_content, fw, indent=4, ensure_ascii=False)
                        except json.JSONDecodeError as e:
                            print("Failed to parse JSON:", e)