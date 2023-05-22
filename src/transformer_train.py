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
from lib import transformer_tools, lstm_tools
import argparse
import logging
import os

# from clearml import Task
# task = Task.init(project_name="ARIMA-MRRNN", task_name="ARIMA-Transformer")

logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100)
parser.add_argument('--batch', default=128)
parser.add_argument('--model', default='Transformer')
parser.add_argument('--lookback', default=14)
parser.add_argument('--num_transformer_blocks', default=4)
parser.add_argument('--head_size', default=64)
parser.add_argument('--num_heads', default=4)
parser.add_argument('--ff_dim', default=4)
args = parser.parse_args()

today = lstm_tools.get_today()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_ARIMA/')
if not os.path.exists(os.path.join(parent_dir, f'models/{str(today)}')):
    os.makedirs(os.path.join(parent_dir, f'models/{str(today)}'))
    os.makedirs(os.path.join(parent_dir, f'out/{args.model}_error/{str(today)}'))
    os.makedirs(os.path.join(parent_dir, f'out/{args.model}_plot/{str(today)}'))
    os.makedirs(os.path.join(parent_dir, f'out/hybrid_model_error/{str(today)}'))
    os.makedirs(os.path.join(parent_dir, f'out/hybrid_model_plot/{str(today)}'))

data = []
d = 0

for file in os.listdir(files_dir):
    file_path = os.path.join(files_dir, file)
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    if not d:
        dates = df.index
        d = 1
    if np.isnan(df).any().any() or np.isnan(df).any().any() :
        continue
    data.append(df.reset_index(drop=True))
    
# scaler = MinMaxScaler()
X, y, train_X, train_y, val_X, val_y, test_X, test_y = [],[],[],[],[],[],[],[]
lookback = args.lookback
future_n = 1

for pair_corr in data:
    # scaler.fit(pair_corr)
    # pair_corr = scaler.transform(pair_corr)
    for i in range(lookback, len(pair_corr) - future_n + 1):
        X.append(pair_corr[i - lookback:i])
        y.append(pair_corr[i + future_n - 1:i + future_n]['close'])

X = np.array([X[i].values.tolist() for i in range(len(X))])
y = np.array([a.values.tolist() for a in y])
train_X, train_y = X[:int(len(X) * 0.85)], y[:int(len(X) * 0.85)]
val_X, val_y = X[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))], y[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))]
test_X, test_y = X[int(len(X) * (0.7 + 0.15)):], y[int(len(X) * (0.7 + 0.15)):]
input_shape = (train_X.shape[1], train_X.shape[2])

# train
tr = transformer_tools.Transformer(14, 8, 1, int(args.num_transformer_blocks), int(args.head_size), int(args.num_heads), int(args.ff_dim))
history = tr.train(train_X, train_y, today, int(args.epochs), int(args.batch))
tr.visualize_loss_plot(history, today)
best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/transformer_{str(args.num_transformer_blocks)}_{str(args.head_size)}_{str(args.num_heads)}_{str(args.ff_dim)}.h5'))
tr.evaluate_model(best_model, train_X, train_y, val_X, val_y, test_X, test_y, today)

_, mse_result, mae_result,_ = tr.evaluate(test_X, test_y)
print("MSE:", mse_result, "MAE:", mae_result)