import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from utils import rnn_tools
import argparse
import logging
import os
from clearml import Task

logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--cell', default='LSTM')
parser.add_argument('--epochs', default=100)
parser.add_argument('--batch', default=64)
parser.add_argument('--neurons', default=32, help="number of neurons")
parser.add_argument('--double_layer', default=0, help="is double layered")
parser.add_argument('--dropout', default=0.5, help="Dropout Rate")
parser.add_argument('--lr', default=0.001, help="learning rate")
parser.add_argument('--lookback', default=14)
args = parser.parse_args()

task = Task.init(project_name="ARIMA-MRRNN", task_name=f"ARIMA-{args.cell} lookback={args.lookback}")
today = rnn_tools.get_today()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_ARIMA_full/')
sorted_dir = os.listdir(files_dir)

# if not os.path.exists(os.path.join(parent_dir, f'models/{str(today)}')):
#     os.makedirs(os.path.join(parent_dir, f'models/{str(today)}'))
#     os.makedirs(os.path.join(parent_dir, f'out/hybrid_model_error/{str(today)}'))
#     os.makedirs(os.path.join(parent_dir, f'out/hybrid_model_plot/{str(today)}'))
#     os.makedirs(os.path.join(parent_dir, f"out/MRRNN_error/{str(today)}"))
#     os.makedirs(os.path.join(parent_dir, f"out/MRRNN_plot/{str(today)}"))


data = []
d = 0

for file in sorted_dir:
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
lookback = int(args.lookback)
future_n = 1

for pair_corr in data:
    # scaler.fit(pair_corr)
    # pair_corr = scaler.transform(pair_corr)
    for i in range(lookback, len(pair_corr) - future_n + 1):
        X.append(pair_corr[i - lookback:i])
        y.append(pair_corr[i + future_n - 1:i + future_n]['close'])

X = np.array([X[i].values.tolist() for i in range(len(X))])
y = np.array([a.values.tolist() for a in y])
train_X, train_y = X[:int(len(X) * 0.7)], y[:int(len(X) * 0.7)]
val_X, val_y = X[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))], y[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))]
test_X, test_y = X[int(len(X) * (0.7 + 0.15)):], y[int(len(X) * (0.7 + 0.15)):]
input_shape = (train_X.shape[1], train_X.shape[2])

# print(f"size:{train_X.shape}")
# print(val_X.shape)
# print(test_X.shape)

# train
model = rnn_tools.rnn_model(int(args.double_layer), int(args.neurons), float(args.dropout), float(args.lr), cell=args.cell)
rnn_model = model.build(input_shape, 1)
checkpoint_dir = os.path.join(parent_dir, f'models/{str(today)}/{args.cell}_{str(args.double_layer)}_{str(args.neurons)}_{str(args.dropout)}_{str(args.lr)}.h5')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint_callback = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, 
                                      mode='min', save_best_only=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

history = rnn_model.fit(train_X, train_y, epochs=int(args.epochs), batch_size=int(args.batch), validation_data=(val_X, val_y), callbacks=[checkpoint_callback, lr_reduction, early_stopping])

if not os.path.exists(os.path.join(parent_dir, f'out/{args.cell}_error/{str(today)}')):
    os.makedirs(os.path.join(parent_dir, f'out/{args.cell}_error/{str(today)}'))
    os.makedirs(os.path.join(parent_dir, f'out/{args.cell}_plot/{str(today)}'))
model.visualize_loss_plot(history, today)
model.evaluate_model(train_X, train_y, val_X, val_y, test_X, test_y, today)