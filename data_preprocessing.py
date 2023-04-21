import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import functools as ft
import itertools
import argparse
from libarary import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--drop', default=0.01, help='dropping threshold')
parser.add_argument('--window', default=100, help="rolling window")
args = parser.parse_args()

# 合併表格
raw_data_path = os.path.join(os.path.abspath(''), 'data/raw_data')
etf50_list = pd.read_csv(raw_data_path + '/etf50_tickers.csv', encoding='Big5').stock_id.tolist()
subdir_dict = {'Major_Investor':'mii', 'Margin_Short_Sell':'mtss', 'Technical':'technical'}
stock_list = {}

for ticker in etf50_list:
    subfile_list = []
    for subdir in  subdir_dict.keys():
        subfile_path = raw_data_path + f'/{subdir}/{ticker}_{subdir_dict[subdir]}.csv' 
        subfile_df = pd.read_csv(subfile_path, index_col='date', parse_dates=True)
        subfile_df = subfile_df.sort_values('date')
        subfile_list.append(subfile_df)

    ticker_df = ft.reduce(lambda left, right:pd.merge(left, right, left_on=['date','stock_id'], right_on = ['date','stock_id']), subfile_list)
    stock_list[ticker] = ticker_df
    
# 填補缺失值、刪除 stock_id 欄位
stock_list, removed_list = preprocess.removeNAvalue(stock_list, args.drop)
for key in stock_list.keys():
    stock_list[key] = stock_list[key].drop('stock_id', axis=1)

# 移動窗口取相關係數
window_size = args.window
comb = list(itertools.combinations(stock_list.keys(), 2))
for (ticker1, ticker2) in comb:
    corr = pd.DataFrame(stock_list[ticker1].rolling(window_size).corr(stock_list[ticker2]))
    corr.drop(corr.index[:100], inplace=True)
    corr.to_csv(f'./data/preprocessed_data/{ticker1}_{ticker2}.csv', index_label='date')