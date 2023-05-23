import pandas as pd
import numpy as np
import datetime

def getNAratio(stock_list):
    NAratio = {}
    ticker_length_list = [len(stock_list[ticker]) for ticker in stock_list.keys()]
    for ticker in stock_list.keys():
        NAratio[ticker] = 1 - (len(stock_list[ticker]) / max(ticker_length_list))
    return NAratio

def findNAratio0(NAratio_dict):
    for (ticker, NAratio) in NAratio_dict.items():
        if NAratio == 0:
            return ticker

def removeNAvalue(stock_list, removed_threshold):
    NAratio_dict = getNAratio(stock_list)
    NAratio0_ticker = findNAratio0(NAratio_dict)
    removed_list = []
    # check_missing_df = {key: value for key, value in NAratio_dict.items() if value > 0 and value < removed_threshold}
    for (ticker, NAratio) in NAratio_dict.items():
        if NAratio > removed_threshold:
            removed_list.append(ticker)
            del stock_list[ticker]
        elif NAratio > 0:
            for date_index in stock_list[NAratio0_ticker].index.difference(stock_list[ticker].index):
                num_columns = len(stock_list[ticker].columns)
                new_row = pd.DataFrame([[None] * num_columns],
                                    columns=stock_list[ticker].columns,
                                    index=[pd.Timestamp(date_index)])
                stock_list[ticker] = pd.concat([stock_list[ticker].loc[:date_index], 
                                                new_row, 
                                                stock_list[ticker].loc[date_index+ datetime.timedelta(days=1):]])
                stock_list[ticker] = stock_list[ticker].fillna(method='ffill')
    return stock_list, removed_list