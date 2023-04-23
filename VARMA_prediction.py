import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random

from statsmodels.tsa.arima_model import ARIMA, ARIMAResults, ARMA, ARMAResults
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
parser.add_argument('--testratio',  default=0.2, help='test ratio')
args = parser.parse_args()

# 讀取檔案
preprocessed_path = './data/preprocessed_data'
corr_matrix = pd.read_csv(os.path.join(preprocessed_path, args.filename), index_col='date', parse_dates=True)
corr_matrix = corr_matrix.dropna()

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

transformed_matrix = corr_matrix.copy()
for i in range(max_diff):
    transformed_matrix = transformed_matrix.diff().dropna()

# 資料分割
nobs = int(len(transformed_matrix) * args.testratio)
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

# varma
train_model = VARMAX(train, order=preferred_order, trend='c')
result = train_model.fit(maxister=1000, disp=False)
test_forecast = arima_tools.test_set_forecast(result, test) # 測試資料集
train_predict = arima_tools.train_set_predict(result, train) #訓練資料集
if max_diff > 0 :
    test_forecast = arima_tools.indiff(test_forecast, corr_matrix, max_diff, 1, nobs)
    train_predict = arima_tools.indiff(train_predict, corr_matrix, max_diff, 0, nobs)

# evaluation
df_error = pd.DataFrame(columns=['VARMA test RMSE', 'VARMA train RMSE', 'mean', 'VARMA RMSE'])
for col in test_forecast.columns:
    testRMSE = rmse(corr_matrix[col].iloc[-nobs:], test_forecast[col])
    trainRMSE = rmse(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], train_predict[col])
    df_error.loc[col, 'VARMA test RMSE'] = testRMSE
    df_error.loc[col, 'VARMA train RMSE'] = trainRMSE
    df_error.loc[col, 'mean'] = corr_matrix[col].mean()

# 套用序列資料
model = VARMAX(train, order=preferred_order, trend='c')
result = model.fit(maxister=1000, disp=False)
prediction = arima_tools.train_set_predict(result, transformed_matrix) # 整段序列資料
if max_diff > 0 :
    prediction = arima_tools.indiff(prediction, corr_matrix, max_diff, 0, nobs)

for col in prediction.columns:
    wholeRMSE = rmse(corr_matrix[col].iloc[max_diff:], prediction[col])
    df_error.loc[col, 'VARMA RMSE'] = wholeRMSE

# 生成圖表
random.seed()
if random.random() < 0.05:
    arima_tools.VARMA_visualize(corr_matrix, prediction, '2330', '1101', preferred_order, 0, 0, nobs)