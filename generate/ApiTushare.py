import json
import tushare as ts
import pandas as pd
from typing import Optional

'''
Todo:
    1. bing的结果没有办法限制时间
    2. 对比有无工具调用的结果
    3. 给出一个当前最好模型的实现链路

'''

class ApiTushare(object):
    def __init__(self):
        self.pro = ts.pro_api()
        with open('query_base/column_name_explanation.json', 'r') as f:
            self.column_name_explanation = json.load(f)


    def __stock_name(self, ts_code: str) -> Optional[str]:
        '''
        股票名字获取
        :param ts_code: 股票代码
        :return 对应的股票名称。如果返回为None，则没有查找到
        '''
        # 获取当前股票的基本信息，包含股票名称
        df_name = self.pro.stock_basic(ts_code=ts_code)
        print(df_name)
        if len(df_name) == 0:
            return None
        return df_name.iloc[0]['name']


    def __stock_ts_code(self, name: str) -> Optional[str]:
        '''
        股票代码获取
        :param name: 股票名称
        :return 对应的股票代码。如果返回为None，则没有查找到
        '''
        # 获取当前股票的基本信息，包含股票代码 ts_code
        df_name = self.pro.stock_basic(name=name)
        print(df_name)
        if len(df_name) == 0:
            return None
        return df_name.loc[0, 'ts_code']


    def stock_price_company(self, name: str, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        公司股价获取
        对应的模版: 个股分析/个股数据查询/[时间][公司名称][财务指标]查询，对应股价
        :param name: 股票名称
        :param trade_date：交易日
        :return 对应的Dataframe。如果返回为None，则没有查找到
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        # 沪深股票/行情数据/日线行情
        # 股价，ts_code为股票代码
        df = self.pro.daily(ts_code=ts_code, start_date=trade_date, end_date=trade_date)
        return df, self.column_name_explanation[self.stock_price_company.__name__]


    def stock_price_company_highest_price(self, name: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        某公司的历史最高股价
        对应的模版: 个股分析/个股数据查询/[公司名称][时间][财务指标]查询，对应历史最高估价
        :param name: 公司名称
        :return 对应的Dataframe。如果返回为None，则没有查找到
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.daily(ts_code=ts_code)
        if len(df) == 0:
            return None, None
        df_max = df[df['high'] == df['high'].max()]
        return df_max, self.column_name_explanation[self.stock_price_company_highest_price.__name__]


    def stock_price_basic(self, name: str, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        公司基本指标获取
        对应的模版: 个股分析/个股数据查询/[公司名称][时间][财务指标]查询，其中：“财务指标”为股息率，市净率，市盈率，换手率
        :param name: 股票名称
        :param trade_date: 交易日
        :return 对应的Dataframe。如果返回为None，则没有查找到
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        # 沪深股票/行情数据/日线行情
        # 股价，ts_code为股票代码
        df = self.pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields='turnover_rate,pe,pb,ps,dv_ratio,total_mv')
        return df, self.column_name_explanation[self.stock_price_basic.__name__]


    def stock_price_daily_largest_increase(self, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        单日涨幅最多的股票的获取
        对应的模版: 市场分析/市场数据查询/[时间][市场类型]涨幅最多的是哪只股票
        :param trade_date: 交易日
        :return 对应的Dataframe，包含股票名称列name。如果返回为None，则没有查找到
        '''
        df = self.pro.daily(trade_date=trade_date)
        if len(df) == 0:
            return None, None
        df_max = df[df['change'] == df['change'].max()]
        df_max['name'] = df_max['ts_code'].apply(self.__stock_name)
        return df_max, self.column_name_explanation[self.stock_price_daily_largest_increase.__name__]


    def stock_price_daily_largest_drop(self, start_date: str, end_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        跌幅最多的股票的获取
        对应的模版: 市场分析/市场数据查询/[股票名称][时间]最大的单日跌幅是多少
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return 对应的Dataframe，包含股票名称列name。如果返回为None，则没有查找到
        '''
        df = self.pro.daily(start_date=start_date, end_date=end_date)
        if len(df) == 0:
            return None, None
        df_min = df[df['pct_chg'] == df['pct_chg'].min()]
        df_min['name'] = df_min['ts_code'].apply(self.__stock_name)
        return df_min, self.column_name_explanation[self.stock_price_daily_largest_drop.__name__]


    def stock_price_daily_highest_open_price(self, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        单日开盘价最高的股票获取
        对应的模版: 市场分析/市场数据查询/[时间][市场类型]开盘价最高的是哪只股票
        :param trade_date: 交易日
        :return 对应的Dataframe，包含股票名称列name。如果返回为None，则没有查找到
        '''
        df = self.pro.daily(trade_date=trade_date)
        if len(df) == 0:
            return None, None
        df_max = df[df['open'] == df['open'].max()]
        df_max['name'] = df_max['ts_code'].apply(self.__stock_name)
        return df_max, self.column_name_explanation[self.stock_price_daily_highest_open_price.__name__]
        # df_max = df[df['open'] > 200]
        # df_max['name'] = df_max['ts_code'].apply(self.__stock_name)
        # return df_max, self.column_name_explanation[self.stock_price_daily_highest_open_price.__name__]


    def stock_price_daily_turnover_difference(self, current_trade_date: str, prior_trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        成交量同比指定交易日下降的数量
        对应的模版: 市场分析/市场数据查询/[时间1]的[市场类型]成交量同比[时间2]下降了多少
        :param current_trade_date: 当前交易日
        :param prior_trade_date: 指定之前的交易日
        :return 成交量的差
        '''
        df_current = self.pro.daily(trade_date=current_trade_date)
        df_prior = self.pro.daily(trade_date=prior_trade_date)
        if len(df_current) == 0 or len(df_prior) == 0:
            return None, None
        return pd.DataFrame({'difference': df_current['vol'].sum() - df_prior['vol'].sum()}, index=[0]), self.column_name_explanation[self.stock_price_daily_turnover_difference.__name__]


    def stock_price_monthly_highest_close_price(self, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        月收盘价最高的股票获取
        对应的模版: 市场分析/市场数据查询/[时间][市场类型]月收盘价最高的是哪只股票
        :param trade_date: 交易月（每月最后一个交易日日期）
        :return 对应的Dataframe，包含股票名称列name。如果返回为None，则没有查找到
        '''
        df = self.pro.monthly(trade_date=trade_date)
        if len(df) == 0:
            return None, None
        df_max = df[df['close'] == df['close'].max()]
        df_max['name'] = df_max['ts_code'].apply(self.__stock_name)
        return df_max, self.column_name_explanation[self.stock_price_monthly_highest_close_price.__name__]


    def stock_limit(self, name: str, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        一个公司在交易日的涨跌停价格
        对应的模版: 市场分析/市场数据查询/[时间][股票名称][涨/跌]停价格
        :param name: 股票名称
        :param trade_date: 交易日
        :return: 涨跌停对应的Dataframe
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.stk_limit(ts_code=ts_code, trade_date=trade_date)
        if len(df) == 0:
            return None, None
        return df, self.column_name_explanation[self.stock_limit.__name__]

    def company_former_names(self, name: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查找一个公司的曾用名
        对应的模版: 个股分析/个股数据查询/[公司名称]的曾用名有哪些
        API的问题：会把所有行都显示两次
        :param name: 公司的现用名
        :return: 对应的Dataframe，name列为其用过的名字，包含现用名
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.namechange(ts_code=ts_code, fields='ts_code,name,start_date,end_date,change_reason')
        return df, self.column_name_explanation[self.company_former_names.__name__]


    def stock_statistics_income(self, name: str, period: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询某公司的利润相关数据
        对应的模版: 个股分析/个股数据查询/[时间][公司名称][财务指标]查询，其中，“财务指标”有净利润，营业总收入，营业利润，营业总成本
        :param name: 公司名
        :param period: 阶段末尾（比如20171231表示年报，20170630半年报，20170930三季报）
        :return: 对应的Dataframe
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.income(ts_code=ts_code, period=period, fields='ts_code,n_income,ann_date,end_date,total_revenue,total_cogs,operate_profit')
        return df, self.column_name_explanation[self.stock_statistics_income.__name__]


    def stock_statistics_market_value(self, name: str, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询某公司的总市值
        对应的模版: 个股分析/个股数据查询/[时间][公司名称][财务指标]查询，其中，“财务指标”为成交量，成交额，开盘价，收盘价
        :param name: 公司名
        :param trade_date: 交易日
        :return: 对应的Dataframe
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.bak_daily(ts_code=ts_code, trade_date=trade_date, fields='trade_date,ts_code,name,close,open,vol,amount,pct_change,interval_3')
        return df, self.column_name_explanation[self.stock_statistics_market_value.__name__]


    def stock_statistics_financial_indicator(self, name: str, period: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询某公司的利润相关数据
        对应的模版: 个股分析/个股数据查询/[时间][公司名称][财务指标]查询，其中，“财务指标”有净债务，销售净利率(单季度)，销售毛利率(单季度)，归属母公司股东的净利润同比增长率，每股营业总收入
                个股分析/财务数据分析/[公司名称][时间]债务状况
        :param name: 公司名
        :param period: 阶段末尾（比如20171231表示年报，20170630半年报，20170930三季报）
        :return: 对应的Dataframe
        '''
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None, None
        df = self.pro.fina_indicator(ts_code=ts_code, period=period, fields='=ts_code,ann_date,netdebt,q_netprofit_margin,q_gsprofit_margin,netprofit_yoy,total_revenue_ps')
        return df, self.column_name_explanation[self.stock_statistics_financial_indicator.__name__]


    def china_cpi(self, month: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询中国的CPI数据
        对应的模版: 宏观分析/经济指标解读/[时间]查询[经济指标]，其中“经济指标”为CPI
        :param month: 月份（YYYYMM）
        :return: 对应的Dataframe
        '''
        df = self.pro.cn_cpi(start_m=month, end_m=month)
        if len(df) == 0:
            return None, None
        return df, self.column_name_explanation[self.china_cpi.__name__]


    def china_ppi(self, month: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询中国的PPI数据
        对应的模版: 宏观分析/经济指标解读/[时间]查询[经济指标]，其中“经济指标”为PPI
        :param month: 月份（YYYYMM）
        :return: 对应的Dataframe
        '''
        df = self.pro.cn_ppi(start_m=month, end_m=month)
        if len(df) == 0:
            return None, None
        return df, self.column_name_explanation[self.china_ppi.__name__]


    def china_gdp(self, quarter: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        查询中国的GDP数据
        对应的模版: 宏观分析/经济指标解读/[时间]查询[经济指标]，其中“经济指标”为GDP，第一产业累计值等
        :param quarter: 季度（2019Q1表示，2019年第一季度）
        :return: 对应的Dataframe
        '''
        df = self.pro.cn_gdp(start_q=quarter, end_q=quarter)
        if len(df) == 0:
            return None, None
        return df, self.column_name_explanation[self.china_gdp.__name__]


    def fund_data(self, name: str, management: str) -> (Optional[pd.DataFrame], Optional[dict]):
        '''
        基金数据查询
        对应的模版: 基金分析/基金数据查询/[管理方]的[基金名称]的[数据]，其中“数据”为成立日期，上市时间，发行日期
        :param name: 基金名称
        :param management: 管理方
        :return: 对应的Dataframe
        '''
        df = self.pro.fund_basic(fields='ts_code,name,management,found_date,list_date,issue_date,status,invest_type,type')
        # df_ret = df[df['name']==name]
        df = df[df['management']==management]
        df = df[df['name'].str.contains(name)]
        if len(df) == 0:
            return None, None
        return df, self.column_name_explanation[self.fund_data.__name__]


    # 期权获取 error
    def option_price(self, name: str, trade_date: str) -> (Optional[pd.DataFrame], Optional[dict]):
        ts_code = self.__stock_ts_code(name)
        if ts_code is None:
            return None
        # 沪深股票/行情数据/日线行情
        # 股价，ts_code为股票代码
        df = self.pro.opt_daily(ts_code=ts_code, trade_date=trade_date)
        return df
