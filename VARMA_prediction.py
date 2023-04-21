import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA, ARIMAResults, ARMA, ARMAResults
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse
from libarary import arima_tools

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser()
# parser.add_argument('--filename', required=True, help='Enter filename')
# parser.add_argument('--test-ratio', defualt=0.2, help='Enter filename')
# args = parser.parse_args()

# 讀取檔案
preprocessed_path = './data/preprocessed_data'
corr_matrix = pd.read_csv(os.path.join(preprocessed_path, '2330_2884.csv'), index_col='date', parse_dates=True)
corr_matrix = corr_matrix.dropna()

# 平穩化測試
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
nobs = int(len(transformed_matrix) * 0.2)
train, test = transformed_matrix[0:-nobs], transformed_matrix[-nobs:]

# auto_arima
order_dict = {}
for col in transformed_matrix.columns:
    auto_model = auto_arima(transformed_matrix[col], start_p=0, start_q=0, max_p=6, 
            max_q=6, error_action='ignore', suppress_warnings=True, seasonal=False)
    info =[auto_model.order, auto_model.aic(), auto_model.bic()]
    order_dict[col] = info

# varma
# model = VARMAX(train, order=(4, 0), trend='c')
# result = model.fit(maxiter=1000, disp=False)
# result.summary()