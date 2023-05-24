import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import lstm_tools
import random
import datetime
import argparse
import logging
import os

random.seed()
logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=32, help="number of neurons")
parser.add_argument('--double_layer', default=0, help="is double layered")
parser.add_argument('--dropout', default=0.25, help="Dropout Rate")
parser.add_argument('--lr', default=0.0005, help="learning rate")
parser.add_argument('--lookback', default=14)
args = parser.parse_args()
    
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_ARIMA/')
tmp_dir = os.path.join(parent_dir, 'data/VARMA_ARIMA_prediction/after_ARIMA/')
today = lstm_tools.get_today()
if not os.path.exists(os.path.join(parent_dir, f'models/{str(today)}')):
    today = datetime.datetime.strptime(today, '%m%d') - datetime.timedelta(days=1)
    today = datetime.datetime.strftime(today, '%m%d')
    
hybrid_prediction_mse, hybrid_prediction_mae, arima_prediction_mae, arima_prediction_mse = [], [], [], []
lookback = int(args.lookback)
future_n = 1

# load model
best_lstm_model = lstm_tools.lstm_model(int(args.double_layer), int(args.neurons), float(args.dropout), float(args.lr))
best_lstm_model.restore(today)

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
    
    residual_X, arima_prediction, original, lstm_prediction = [], [], [], []
    for i in range(lookback, len(residual_df) - future_n + 1):
        residual_X.append(residual_df[i - lookback:i])
    residual_X = np.array([residual_X[i].values.tolist() for i in range(len(residual_X))])
    
    lstm_predictions_df = best_lstm_model.predict(residual_X)
    
    for i in range(len(lstm_predictions_df)):
        lstm_prediction.append(lstm_predictions_df[i][0])
    
    arima_prediction = arima_output_df[-len(lstm_prediction):].tolist()
    original = original_df[-len(lstm_prediction):].tolist()
    timestamps = timestamps[-len(lstm_prediction):].tolist()
    hybrid_prediction = [x - y for x, y in zip(arima_prediction, lstm_prediction)]
    
    arima_prediction_mse.append(mean_squared_error(original, arima_prediction))
    arima_prediction_mae.append(mean_absolute_error(original, arima_prediction))
    hybrid_prediction_mse.append(mean_squared_error(original, hybrid_prediction))
    hybrid_prediction_mae.append(mean_absolute_error(original, hybrid_prediction))
    
    if random.random() < 0.05:
        best_lstm_model.visualize_prediction_plot(hybrid_prediction, original, arima_prediction, timestamps, today, file)

df_error = pd.DataFrame(columns=['ARIMA prediction MSE', 'ARIMA prediction MAE', 'ARIMA_LSTM prediction MSE', 'ARIMA_LSTM prediction MAE'])
df_error.loc[0] = lstm_tools.mean(arima_prediction_mse), lstm_tools.mean(arima_prediction_mae), lstm_tools.mean(hybrid_prediction_mse), lstm_tools.mean(hybrid_prediction_mae)
df_error.to_csv(os.path.join(parent_dir, f'out/hybrid_model_error/{str(today)}/ARIMA-LSTM_{str(args.double_layer)}_{str(args.neurons)}_{str(args.dropout)}_{str(args.lr)}.csv'))