import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

data = []
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, 'data/VARMA_ARIMA/filtered_after_ARIMA/')
d = 0

for file in os.listdir(files_dir):
    file_path = os.path.join(files_dir, file)
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    if not d:
        dates = df.index
        d = 1
    if np.isnan(df).any().any() or np.isnan(df).any().any() :
        print(file)
        continue
    data.append(df.reset_index(drop=True))

X, y, train_X, train_y, val_X, val_y, test_X, test_y = [],[],[],[],[],[],[],[]
past_n = 5
future_n = 1
# scaler = MinMaxScaler()

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

model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(8, 'relu'))
model.add(Dense(train_y.shape[1], 'linear'))

model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(train_X, train_y, epochs=10, batch_size=128, validation_data=(val_X, val_y), verbose=1)
