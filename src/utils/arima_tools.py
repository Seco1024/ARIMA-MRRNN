import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults, ARMA, ARMAResults
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from pmdarima.utils import diff_inv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import copy
import os
import warnings

warnings.filterwarnings("ignore")
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC') # .dropna() handles differenced data
    if result[1] <= 0.05:    # Data has no unit root and is stationary
        return True
    else:      # Data has a unit root and is non-stationary
        return False
    
def getOrder(order_dict):
    counter = {}
    for key, lst in order_dict.items():
        if lst[0] not in counter.keys():
            counter[lst[0]] = 1
        else:
            counter[lst[0]] += 1
    max_order = max(counter, key=lambda k:counter[k])
    if max_order != (0, 0, 0):
        return max_order
    else:
        del counter[max_order]
        second_order = max(counter, key=lambda k:counter[k])
        return second_order

def train_set_predict(result, train):
    train_predict = result.predict(start=0, end=len(train)-1, alpha=0.05)
    train_predict = pd.DataFrame(train_predict)
    train_predict['date'] = pd.DatetimeIndex(train.index)
    train_predict.set_index('date', inplace=True)
    return train_predict

def test_set_forecast(result, test):
    test_forecast = result.forecast(len(test))
    test_forecast = pd.DataFrame(test_forecast)
    test_forecast['date'] = pd.DatetimeIndex(test.index)
    test_forecast.set_index('date', inplace=True)
    return test_forecast

def diff(corr_matrix):
    diff_dict = {}
    for col in corr_matrix.columns:
        if col == 'date':
            continue
        differenced_series = corr_matrix[col].copy()
        diff = 0
        isStationary = adf_test(corr_matrix[col])
        while not isStationary:
            diff += 1
            differenced_series = differenced_series.diff().dropna()
            isStationary = adf_test(differenced_series)
        diff_dict[col] = diff
    max_diff = max(diff_dict.values())

    transformed_matrix = corr_matrix.copy()
    for i in range(max_diff):
        transformed_matrix = transformed_matrix.diff().dropna()
        
    return transformed_matrix, max_diff
        
def indiff(forecast, origin, max_diff, isTest, nobs):
    for col in forecast.columns:
        inversed = diff_inv(forecast[col], 1, max_diff)
        if isTest:
            inversed = inversed + origin[col].iloc[-nobs-1]
        else:
            inversed = inversed + origin[col].iloc[max_diff-1]
        forecast[col] = inversed[max_diff:]
    return forecast

def visualize(origin, predict, ticker1, ticker2, model, isTest, isTrain, nobs, order_dict):
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    for col in predict.columns:
        plt.figure()
        if isTest:
            title = f'{model} Prediction on {col} (test):{ticker1}-{ticker2}'
        elif isTrain:
            title = f'{model} Prediction on {col} (train):{ticker1}-{ticker2}'
        else:
            title = f'{model} Prediction on {col} (whole):{ticker1}-{ticker2}'
        ylabel = f'Correlation Coefficient on {col}'
        xlabel=''

        if model == 'ARIMA':
            legend_name = f'{model}{order_dict[col][0]} Prediction'
        elif model == 'VARMA':
            legend_name = f'{model}({getOrder(order_dict)[0]}, {getOrder(order_dict)[2]}) Prediction'

        fcst = predict[col].rename(legend_name)
        ax = fcst.plot(legend=True, figsize=(8,5),title=title)
        origin[col].plot(legend=True)
        ax.autoscale(axis='x',tight=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        label = 'test' if isTest else 'train' if isTrain else 'prediction'
        plt.savefig(os.path.join(parent_dir, f'out/VARMA_ARIMA_plot/after_{model}/{ticker1}-{ticker2}-{label}-{col}.png'))
       
def varma_prediction(corr_matrix, transformed_matrix, order_dict, max_diff, filename): 
    df_error = pd.DataFrame(columns=['ARIMA MSE (test)', 'ARIMA MSE (train)', 'ARIMA MSE(whole)', 'ARIMA MAE (test)', 'ARIMA MAE (train)', 'ARIMA MAE(whole)'])
    nobs = int(len(transformed_matrix) * float(0.2))
    train, test = transformed_matrix[0:-nobs], transformed_matrix[-nobs:]
    varma_dict = copy.deepcopy(order_dict)
    limit = 5
    
    while limit > 0 and varma_dict:
        order = getOrder(varma_dict)
        preferred_order = (order[0], order[2])
        train_model = VARMAX(train, order=preferred_order, trend='c')
        varma_train_result = train_model.fit(maxister=1000, disp=False)
        varma_test_forecast = test_set_forecast(varma_train_result, test)
        varma_train_predict = train_set_predict(varma_train_result, train)
        
        if max_diff > 0 : # 逆差分
            varma_test_forecast = indiff(varma_test_forecast, corr_matrix, max_diff, 1, nobs)
            varma_train_predict = indiff(varma_train_predict, corr_matrix, max_diff, 0, nobs)

        # VARMA evaluation
        for col in varma_test_forecast.columns:
            df_error.loc[col, 'VARMA MSE (test)'] = mean_squared_error(corr_matrix[col].iloc[-nobs:], varma_test_forecast[col])
            df_error.loc[col, 'VARMA MSE (train)'] = mean_squared_error(corr_matrix[col].iloc[max_diff:len(corr_matrix)-nobs], varma_train_predict[col])

        if (df_error.iloc[0,:] > 1.5).any() == True:
            varma_dict = {key: lst for key, lst in varma_dict.items() if lst[0] != (preferred_order[0], 0, preferred_order[1])}
            limit = limit - 1
            continue

        # VARMA re-fit
        varma_model = VARMAX(transformed_matrix, order=preferred_order, trend='c')
        varma_result = varma_model.fit(maxister=1000, disp=False)
        varma_prediction = train_set_predict(varma_result, transformed_matrix)
        
        if max_diff > 0 :
            varma_prediction = indiff(varma_prediction, corr_matrix, max_diff, 0, nobs)

        for col in varma_prediction.columns:
            df_error.loc[col, 'VARMA MSE (whole)'] = mean_squared_error(corr_matrix[col].iloc[max_diff:], varma_prediction[col])

        if (df_error.iloc[0,:] > 1.5).any() == True:
            varma_dict = {key: lst for key, lst in varma_dict.items() if lst[0] != (preferred_order[0], 0, preferred_order[1])}
            limit = limit - 1
        else:
            varma_prediction.to_csv(os.path.join(parent_dir, f'data/VARMA_ARIMA_prediction/after_VARMA/{filename}'))
            limit = 0
            break