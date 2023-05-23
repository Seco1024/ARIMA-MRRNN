import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from pmdarima.utils import diff_inv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import arima_tools
import random
import re
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
preprocessed_path = os.path.join(parent_dir, 'data/preprocessed_data')
parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='Enter filename')
parser.add_argument('--testratio', default=0.2, help='test ratio')
parser.add_argument('--window_size', default=100, help='window size')
args = parser.parse_args()

corr_matrix = pd.read_csv(os.path.join(preprocessed_path, args.filename), index_col='date', parse_dates=True)
corr_matrix = corr_matrix.fillna(method='ffill')

nobs = int(len(corr_matrix) * float(args.testratio))
train, test = corr_matrix[0:-nobs], corr_matrix[-nobs:]

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

# auto_arima
order_dict = {}
for col in corr_matrix.columns:
    auto_model = auto_arima(corr_matrix[col], start_p=0, start_q=0, max_p=6, 
            max_q=6, error_action='ignore', suppress_warnings=True, seasonal=False)
    info =[auto_model.order, auto_model.aic(), auto_model.bic()]
    order_dict[col] = info

# multiple ARIMA
arima_test_forecast = pd.DataFrame(columns=corr_matrix.columns)
arima_train_predict = pd.DataFrame(columns=corr_matrix.columns)
arima_prediction = pd.DataFrame(columns=corr_matrix.columns)
test_start=len(train)
test_end=len(train)+len(test)-1
train_start=train.index[0]
train_end=train.index[len(train)-1]

for col, lst in order_dict.items():
    arima_order = order_dict[col][0]
    arima_train_model = ARIMA(corr_matrix[col].iloc[:-nobs], order=arima_order)
    arima_result = arima_train_model.fit()

    arima_test_forecast.loc[:, col] = arima_result.predict(start=test_start, end=test_end, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Test Forecast')
    arima_train_predict.loc[:, col] = arima_result.predict(start=train_start, end=train_end, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Train Predictions')

arima_test_forecast['date'] = pd.DatetimeIndex(test.index)
arima_test_forecast.set_index('date', inplace=True)
arima_train_predict = arima_train_predict[max_diff:]

# Evaluation
df_error = pd.DataFrame(columns=['ARIMA MSE (test)', 'ARIMA MSE (train)', 'ARIMA MSE(whole)', 'ARIMA MAE (test)', 'ARIMA MAE (train)', 'ARIMA MAE(whole)'])
for col in arima_test_forecast.columns:
    df_error.loc[col, 'ARIMA MSE (test)'] = mean_squared_error(corr_matrix[col].iloc[-nobs:], arima_test_forecast[col])
    df_error.loc[col, 'ARIMA MSE (train)'] = mean_squared_error(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], arima_train_predict[col])
    df_error.loc[col, 'ARIMA MAE (test)'] = mean_absolute_error(corr_matrix[col].iloc[-nobs:], arima_test_forecast[col])
    df_error.loc[col, 'ARIMA MAE (train)'] = mean_absolute_error(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], arima_train_predict[col])

# Multiple ARIMA re-fit
for col, lst in order_dict.items():
    arima_order = order_dict[col][0]
    arima_model = ARIMA(corr_matrix[col], order=arima_order)
    arima_result = arima_model.fit()
    arima_prediction.loc[:, col] = arima_result.predict(start=0, end=len(corr_matrix)-1, dynamic=False, typ='levels').rename(f'ARIMA{arima_order} Prediction')

arima_prediction = arima_prediction[max_diff:]
for col in arima_prediction.columns:
    df_error.loc[col, 'ARIMA MSE(whole)'] = mean_squared_error(corr_matrix[col][max_diff:], arima_prediction[col])
    df_error.loc[col, 'ARIMA MAE(whole)'] = mean_absolute_error(corr_matrix[col][max_diff:], arima_prediction[col])

arima_prediction.to_csv(os.path.join(parent_dir, f'data/VARMA_ARIMA_prediction/after_ARIMA/{args.filename}'))

error_mean = pd.DataFrame(df_error.mean())
error_mean = error_mean.transpose()
if (error_mean.iloc[0,:] > 1.5).any() == True:
    error_mean.to_csv(os.path.join(parent_dir, f'out/VARMA_ARIMA_error/anomalies/{args.filename}'))
    print(f"anomaly residual output for {args.filename}")
else:
    error_mean.to_csv(os.path.join(parent_dir, f'out/VARMA_ARIMA_error/window{args.window_size}/{args.filename}'))

random.seed()
if random.random() < 0.005:
    ticker1, ticker2 = re.findall(r"\d+", args.filename)[0], re.findall(r"\d+", args.filename)[1]
    arima_tools.visualize(corr_matrix, arima_prediction, ticker1, ticker2, 'ARIMA', 0, 0, nobs, order_dict)
    arima_tools.visualize(corr_matrix[-nobs:], arima_test_forecast, ticker1, ticker2, 'ARIMA', 1, 0, nobs, order_dict)
    arima_tools.visualize(corr_matrix[:-nobs], arima_train_predict, ticker1, ticker2, 'ARIMA', 0, 1, nobs, order_dict)

arima_residual_matrix = arima_prediction - corr_matrix[max_diff:]
arima_residual_matrix.to_csv(os.path.join(parent_dir, f"data/VARMA_ARIMA/after_ARIMA/{args.filename}"), index='date')