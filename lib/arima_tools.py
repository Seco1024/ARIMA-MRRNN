import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults, ARMA, ARMAResults
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from pmdarima.utils import diff_inv

import warnings
warnings.filterwarnings("ignore")


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
    for col in predict.columns:
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
        plt.savefig(f'./data/plot/after_{model}/{ticker1}-{ticker2}-{label}-{col}.png')