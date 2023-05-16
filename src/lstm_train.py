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
import argparse
import logging
import os

logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ARIMA', help='ARIMA or VARMA')
parser.add_argument('--neurons', default=64, help="number of neurons")
parser.add_argument('--double_layer', default=0, help="is double layered")
parser.add_argument('--l2', default=1, help="L2 Regularization")
parser.add_argument('--past_n', default=14)
parser.add_argument('--epochs', default=200)
parser.add_argument('--batch', default=128)
args = parser.parse_args()

today = lstm_tools.get_today()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f'data/VARMA_ARIMA/after_{args.model}/')
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
past_n = args.past_n
future_n = 1

for pair_corr in data:
    # scaler.fit(pair_corr)
    # pair_corr = scaler.transform(pair_corr)
    for i in range(past_n, len(pair_corr) - future_n + 1):
        X.append(pair_corr[i - past_n:i])
        y.append(pair_corr[i + future_n - 1:i + future_n]['close'])

X = np.array([X[i].values.tolist() for i in range(len(X))])
y = np.array([a.values.tolist() for a in y])
train_X, train_y = X[:int(len(X) * 0.7)], y[:int(len(X) * 0.7)]
val_X, val_y = X[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))], y[int(len(X) * 0.7):int(len(X) * (0.7 + 0.15))]
test_X, test_y = X[int(len(X) * (0.7 + 0.15)):], y[int(len(X) * (0.7 + 0.15)):]
input_shape = (train_X.shape[1], train_X.shape[2])

# train
model = lstm_tools.build_lstm_model(input_shape, 1, int(args.neurons), 0.5, int(args.double_layer), int(args.l2))
checkpoint_dir = os.path.join(parent_dir, f'models/{str(today)}_{args.model}_{str(args.neurons)}_{str(args.double_layer)}_{str(args.l2)}.h5')
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
checkpoint_callback = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, 
                                      mode='min', save_best_only=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.00005)

history = model.fit(train_X, train_y, epochs=int(args.epochs), batch_size=int(args.batch), validation_data=(val_X, val_y), callbacks=[early_stopping, checkpoint_callback, lr_reduction])
lstm_tools.visualize_loss_plot(history, args.model, args.neurons, args.double_layer, args.l2)
best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}_{args.model}_{str(args.neurons)}_{str(args.double_layer)}_{str(args.l2)}.h5'))
lstm_tools.evaluate_model(best_model, train_X, train_y, val_X, val_y, test_X, test_y, args.model, args.neurons, args.double_layer, args.l2)