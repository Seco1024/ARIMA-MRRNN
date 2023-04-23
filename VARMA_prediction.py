import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import re

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse
from libarary import arima_tools
from pmdarima.utils import diff_inv

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='Enter filename')
parser.add_argument('--testratio', default=0.2, help='test ratio')
args = parser.parse_args()

# 讀取檔案
preprocessed_path = './data/preprocessed_data'
corr_matrix = pd.read_csv(os.path.join(preprocessed_path, args.filename), index_col='date', parse_dates=True)
corr_matrix = corr_matrix.fillna(method='ffill')

# 平穩化
diff_dict = {}
for col in corr_matrix.columns:
    if col == 'date':
        continue
    differenced_series = corr_matrix[col].copy()
    diff = 0
    isStationary = arima_tools.adf_test(corr_matrix[col])
    while not isStationary:
        diff += 1
        differenced_series = differenced_series.diff().dropna()
        isStationary = arima_tools.adf_test(differenced_series)
    diff_dict[col] = diff
max_diff = max(diff_dict.values())
# print(max_diff)

transformed_matrix = corr_matrix.copy()
for i in range(max_diff):
    transformed_matrix = transformed_matrix.diff().dropna()

# 資料分割
nobs = int(len(transformed_matrix) * float(args.testratio))
train, test = transformed_matrix[0:-nobs], transformed_matrix[-nobs:]

# auto_arima
order_dict = {}
for col in transformed_matrix.columns:
    auto_model = auto_arima(transformed_matrix[col], start_p=0, start_q=0, max_p=6, 
            max_q=6, error_action='ignore', suppress_warnings=True, seasonal=False)
    info =[auto_model.order, auto_model.aic(), auto_model.bic()]
    order_dict[col] = info
order = arima_tools.getOrder(order_dict)
preferred_order = (order[0], order[2])

# VARMA
train_model = VARMAX(train, order=preferred_order, trend='c')
varma_train_result = train_model.fit(maxister=1000, disp=False)
varma_test_forecast = arima_tools.test_set_forecast(varma_train_result, test) # 預測：測試資料集
varma_train_predict = arima_tools.train_set_predict(varma_train_result, train) # 預測：訓練資料集
if max_diff > 0 : # 逆差分
    varma_test_forecast = arima_tools.indiff(varma_test_forecast, corr_matrix, max_diff, 1, nobs)
    varma_train_predict = arima_tools.indiff(varma_train_predict, corr_matrix, max_diff, 0, nobs)

# VARMA evaluation
df_error = pd.DataFrame(columns=['mean', 'VARMA RMSE (test)', 'VARMA RMSE (train)', 'VARMA RMSE', 'ARIMA RMSE (test)', 'ARIMA RMSE (train)', 'ARIMA RMSE'])
for col in varma_test_forecast.columns:
    varma_testRMSE = rmse(corr_matrix[col].iloc[-nobs:], varma_test_forecast[col])
    vrama_trainRMSE = rmse(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], varma_train_predict[col])
    df_error.loc[col, 'VARMA RMSE (test)'] = varma_testRMSE
    df_error.loc[col, 'VARMA RMSE (train)'] = vrama_trainRMSE
    df_error.loc[col, 'mean'] = corr_matrix[col].mean()

# VARMA re-fit
varma_model = VARMAX(transformed_matrix, order=preferred_order, trend='c')
varma_result = varma_model.fit(maxister=1000, disp=False)
varma_prediction = arima_tools.train_set_predict(varma_result, transformed_matrix) # 整段序列資料
if max_diff > 0 :
    varma_prediction = arima_tools.indiff(varma_prediction, corr_matrix, max_diff, 0, nobs)

for col in varma_prediction.columns:
    varma_wholeRMSE = rmse(corr_matrix[col].iloc[max_diff:], varma_prediction[col])
    df_error.loc[col, 'VARMA RMSE'] = varma_wholeRMSE

# multiple ARIMA
arima_test_forecast = pd.DataFrame(columns=corr_matrix.columns)
arima_train_predict = pd.DataFrame(columns=corr_matrix.columns)

for col, lst in order_dict.items():
    arima_order = order_dict[col][0]
    arima_train_model = ARIMA(corr_matrix[col].iloc[:-nobs], order=arima_order)
    arima_result = arima_train_model.fit()

    test_start=len(train)
    test_end=len(train)+len(test)-1
    train_start=train.index[0]
    train_end=train.index[len(train)-1]

    arima_col_test_predictions = arima_result.predict(start=test_start, end=test_end, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Test Forecast')
    arima_test_forecast.loc[:, col] = arima_col_test_predictions
    arima_col_train_predictions = arima_result.predict(start=train_start, end=train_end, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Train Predictions')
    arima_train_predict.loc[:, col] = arima_col_train_predictions

arima_test_forecast['date'] = pd.DatetimeIndex(test.index)
arima_test_forecast.set_index('date', inplace=True)

# Multiple ARIMA Evaluation
for col in arima_test_forecast.columns:
    varma_testRMSE = rmse(corr_matrix[col].iloc[-nobs:], arima_test_forecast[col])
    vrama_trainRMSE = rmse(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], arima_train_predict[col])
    df_error.loc[col, 'ARIMA RMSE (test)'] = varma_testRMSE
    df_error.loc[col, 'ARIMA RMSE (train)'] = vrama_trainRMSE

# Multiple ARIMA re-fit
arima_prediction = pd.DataFrame(columns=corr_matrix.columns)

for col, lst in order_dict.items():
    arima_order = order_dict[col][0]
    arima_model = ARIMA(corr_matrix[col], order=arima_order)
    arima_result = arima_model.fit()

    arima_col_prediction = arima_result.predict(start=0, end=len(corr_matrix)-1, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Prediction')
    arima_prediction.loc[:, col] = arima_col_prediction

for col in arima_prediction.columns:
    arima_wholeRMSE = rmse(corr_matrix[col], arima_prediction[col])
    df_error.loc[col, 'ARIMA RMSE'] = arima_wholeRMSE

# 輸出 error
error_mean = pd.DataFrame(df_error.mean())
error_mean = error_mean.transpose()
error_mean.to_csv(f'./data/VARMA_ARIMA/error/{args.filename}')

# 以一定概率生成圖表

random.seed()
if random.random() < 0.05:
    ticker1, ticker2 = re.findall(r"\d+", args.filename)[0], re.findall(r"\d+", args.filename)[1]
    arima_tools.visualize(corr_matrix, varma_prediction, ticker1, ticker2, 'VARMA', 0, 0, nobs)
    arima_tools.visualize(corr_matrix, arima_prediction, ticker1, ticker2, 'ARIMA', 0, 0, nobs)
    arima_tools.visualize(corr_matrix[-nobs:], varma_test_forecast, ticker1, ticker2, 'VARMA', 1, 0, nobs)
    arima_tools.visualize(corr_matrix[-nobs:], arima_test_forecast, ticker1, ticker2, 'ARIMA', 1, 0, nobs)
    arima_tools.visualize(corr_matrix[:-nobs], varma_train_predict, ticker1, ticker2, 'VARMA', 0, 1, nobs)
    arima_tools.visualize(corr_matrix[:-nobs], arima_train_predict, ticker1, ticker2, 'ARIMA', 0, 1, nobs)