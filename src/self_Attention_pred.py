import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import rnn_tools, self_Attention_tools
import random
import datetime
import argparse
import logging
import os

random.seed()
logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--num_transformer_blocks', default=4)
parser.add_argument('--head_size', default=64)
parser.add_argument('--num_heads', default=4)
parser.add_argument('--ff_dim', default=4)
parser.add_argument('--dropout', default=0.25, help="Dropout Rate")
parser.add_argument('--lr', default=0.0005, help="learning rate")
parser.add_argument('--lookback', default=14)
args = parser.parse_args()
    
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_ARIMA/')
tmp_dir = os.path.join(parent_dir, 'data/VARMA_ARIMA_prediction/after_ARIMA/')
today = rnn_tools.get_today()
if not os.path.exists(os.path.join(parent_dir, f'models/{str(today)}')):
    today = datetime.datetime.strptime(today, '%m%d') - datetime.timedelta(days=1)
    today = datetime.datetime.strftime(today, '%m%d')
    
hybrid_prediction_mse, hybrid_prediction_mae, arima_prediction_mae, arima_prediction_mse = [], [], [], []
lookback = int(args.lookback)
future_n = 1

# load model
best_sa_model = self_Attention_tools.self_Attention(lookback, 8, float(args.dropout), float(args.lr), int(args.num_transformer_blocks), int(args.head_size), int(args.num_heads), int(args.ff_dim))
best_sa_model.restore(today)

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
    
    residual_X, arima_prediction, original, sa_prediction = [], [], [], []
    for i in range(lookback, len(residual_df) - future_n + 1):
        residual_X.append(residual_df[i - lookback:i])
    residual_X = np.array([residual_X[i].values.tolist() for i in range(len(residual_X))])
    
    sa_predictions_df = best_sa_model.predict(residual_X)
    
    for i in range(len(sa_predictions_df)):
        sa_prediction.append(sa_predictions_df[i][0])
    
    arima_prediction = arima_output_df[-len(sa_prediction):].tolist()
    original = original_df[-len(sa_prediction):].tolist()
    timestamps = timestamps[-len(sa_prediction):].tolist()
    hybrid_prediction = [x - y for x, y in zip(arima_prediction, sa_prediction)]
    
    arima_prediction_mse.append(mean_squared_error(original, arima_prediction))
    arima_prediction_mae.append(mean_absolute_error(original, arima_prediction))
    hybrid_prediction_mse.append(mean_squared_error(original, hybrid_prediction))
    hybrid_prediction_mae.append(mean_absolute_error(original, hybrid_prediction))
    
    if random.random() < 0.05:
        best_sa_model.visualize_prediction_plot(hybrid_prediction, original, arima_prediction, timestamps, today, file)

df_error = pd.DataFrame(columns=['ARIMA prediction MSE', 'ARIMA prediction MAE', 'ARIMA_self_Attention prediction MSE', 'ARIMA_self_Attention prediction MAE'])
df_error.loc[0] = rnn_tools.mean(arima_prediction_mse), rnn_tools.mean(arima_prediction_mae), rnn_tools.mean(hybrid_prediction_mse), rnn_tools.mean(hybrid_prediction_mae)
df_error.to_csv(os.path.join(parent_dir, f'out/hybrid_model_error/{str(today)}/ARIMA-self_Attention_{args.num_transformer_blocks}_{args.head_size}_{args.num_heads}_{args.ff_dim}_{args.dropout}_{args.lr}.csv'))