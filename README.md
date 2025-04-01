## python脚本文件介绍
### 数据集构建流程
1. [QueryIntentClassifier.py](generate/QueryIntentClassifier.py): 用LLM标注已有query的意图，输出与输出均在data文件夹
2. 提取查询类（query base），并将第一步的结果转化为[query_base](query_base/query_base.json)的格式
3. [TextRetriever.py](generate/TextRetriever.py): 从query_base文档中找到query，并在数据库中检索到相关文档，存放到rel_docs对应的的文件夹下
4. [ApiRetriever.py](generate/ApiRetriever.py): 调用Tushare API，将API相关数据存放到rel_docs对应的的文件夹下
5. 根据第3步中检索文档的格式，将其转化为仅包含文档的列表格式，命名为merge.json
6. [RelevanceScorer.py](generate/RelevanceScorer.py): 为每个query对应的merge.json中的文档使用LLM打分
7. [LowRelFilter.py](generate/LowRelFilter.py): 只保留相关分数高于lower_bound的文本，默认为6分。输入为score.json，输出为score_rel.json。score_rel_6.json为高于6分的所有文本，作为保留；score_rel.json会手动删去不相关的其他文本
8. [Generator.py](generate/Generator.py): 生成回答，使用openAI调用方式
9. [Cluster.py](generate/Cluster.py): 把各个query及其回答汇总到两个文件当中（分为content型和value型），文件名与本次回答的大模型相关
10. 关于评价的脚本文件均在[evaluate](evaluate)文件夹下：[basic.py](evaluate/basic.py)完成value类accuracy的评价和content类rouge-l的评价；[llm_pairwise.py](evaluate/llm_pairwise.py)完成content类需要成对比较指标的评价；[evaluate/llm_pointwise.py](evaluate/llm_pointwise.py)完成content类需要直接比较指标的评价

### 其他脚本文件
[ApiTushare.py](utils/ApiTushare.py): 需要调用的API\
[base2intents.py](utils/base2intents.py): 将query_base文件转化为intents，便于观察\
[base2queries.py](utils/base2queries.py): 将query_base文件转化为queries，便于观察\
[cal_ratio.py](utils/cal_ratio.py): 计算6分以上文档的相关性比率，输出为result.json\
[list2dict.py](utils/list2dict.py): 将v1版query_base转化为v2，即query级别为字典，其键为query，值为请求时的参数\
[score_distribution.py](utils/score_distribution.py): 统计score.json的分数分布情况，输出为distribution.json
[base_with_dis.py](utils/base_with_dis.py): 为query_base的query添加相关文本的数量字段，便于进行查看，输出为base_with_query.json
[json2excel.py](utils/json2excel.py): 将distribution.json转化为excel，方便统计
[delete_file.py](utils/delete_file.py): 删除三级文件夹下对应文件名的文件，方便调试（因为之前的代码加了判断一个文件是否存在的“断点逻辑”


## 数据介绍
### 相关文档[rel_docs](rel_docs)
v{版本号}为文件夹的名称，其各级文件夹名分别为一级意图、二级意图、query，query文件夹下的文件分别为
1. news/report/wechat_{recallnum}_{size}.json: 检索到的相关文档，类型分别为news/report/wechat，召回参数和精排参数分别为recallnum和size
2. merge.json: 将news/report/wechat相关文档合并
3. score.json: 使用大模型对每个相关文档进行打分
4. score_rel_6: 保留6分及以上的文本
5. score_rel: 保留6分及以上的文本，并手动删除不相关的文本
v{版本号}为文件夹下还有两个文件：
distribution.json: 统计所有query的分数分布\
result.json: 统计每个query真实相关文本和6分级以上文本数量的比值
6. json2excel.py: 将distribution.json转化为excel，方便统计