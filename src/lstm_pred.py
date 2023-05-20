import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import regularizers, metrics, backend
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lib import lstm_tools
import random
import datetime
import argparse
import logging
import os

logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed()
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='LSTM')
parser.add_argument('--neurons', default=64, help="number of neurons")
parser.add_argument('--double_layer', default=0, help="is double layered")
parser.add_argument('--l2', default=0.01, help="L2 Regularization")
parser.add_argument('--dropout', default=0.5, help="Dropout Rate")
parser.add_argument('--past_n', default=14)
args = parser.parse_args()
    
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_ARIMA/')
tmp_dir = os.path.join(parent_dir, 'data/VARMA_ARIMA_prediction/after_ARIMA/')

today = lstm_tools.get_today()
if not os.path.exists(os.path.join(parent_dir, f'models/{str(today)}')):
    today = lstm_tools.get_today()
    today = datetime.strptime(today, '%m%d') - datetime.timedelta(days=1)
    today = datetime.strftime(today, '%m%d')

best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/{args.model}_{str(args.neurons)}_{str(args.double_layer)}_{str(args.l2)}.h5'))
hybrid_prediction_mse, hybrid_prediction_mae, arima_prediction_mae, arima_prediction_mse = [], [], [], []
past_n = args.past_n
future_n = 1

for file in os.listdir(tmp_dir):
    residual_file_path = os.path.join(files_dir, file)
    residual_df = pd.read_csv(residual_file_path, index_col='date', parse_dates=True)
    residual_df = residual_df.reset_index(drop=True)
    arima_output_path = os.path.join(parent_dir, f'data/VARMA_ARIMA_prediction/after_ARIMA/{file}')
    arima_output_df = pd.read_csv(arima_output_path, index_col='date', parse_dates=True)
    arima_output_df = arima_output_df.reset_index(drop=True)
    arima_output_df = arima_output_df['close']
    original_path = os.path.join(parent_dir, f'data/preprocessed_data/{file}')
    original_df = pd.read_csv(original_path, index_col='date', parse_dates=True)
    timestamps = original_df.index
    original_df = original_df.reset_index(drop=True)
    original_df = original_df['close']
    original_df = original_df[-len(arima_output_df):]
    
    residual_X, arima_prediction, original, lstm_predict = [], [], [], []
    for i in range(past_n, len(residual_df) - future_n + 1):
        residual_X.append(residual_df[i - past_n:i])
    residual_X = np.array([residual_X[i].values.tolist() for i in range(len(residual_X))])
    predictions_df = best_model.predict(residual_X)
    for i in range(len(predictions_df)):
        lstm_predict.append(predictions_df[i][0])
    
    arima_prediction = arima_output_df[-len(lstm_predict):].tolist()
    original = original_df[-len(lstm_predict):].tolist()
    timestamps = timestamps[-len(lstm_predict):].tolist()
    hybrid_prediction = [x - y for x, y in zip(arima_prediction, lstm_predict)]
    
    arima_prediction_mse.append(mean_squared_error(original, arima_prediction))
    arima_prediction_mae.append(mean_absolute_error(original, arima_prediction))
    hybrid_prediction_mse.append(mean_squared_error(original, hybrid_prediction))
    hybrid_prediction_mae.append(mean_absolute_error(original, hybrid_prediction))
    
    if random.random() < 0.002:
        lstm_tools.visualize_prediction_plot(hybrid_prediction, original, timestamps, args.model, args.neurons, args.double_layer, args.l2, file, today)

df_error = pd.DataFrame(columns=['ARIMA prediction MSE', 'ARIMA prediction MAE', 'ARIMA_LSTM prediction MSE', 'ARIMA_LSTM prediction MAE'])
df_error.loc[0] = lstm_tools.mean(arima_prediction_mse), lstm_tools.mean(arima_prediction_mae), lstm_tools.mean(hybrid_prediction_mse), lstm_tools.mean(hybrid_prediction_mae)
df_error.to_csv(os.path.join(parent_dir, f'out/hybrid_model_error/{str(today)}/{str(args.model)}/{str(args.neurons)}_{str(args.double_layer)}_{str(args.l2)}'))