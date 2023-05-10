import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error
import os

def build_lstm_model(x_shape, y_shape, neurons, dropout_rate, is_doubled_layer, is_regularized):
    model = Sequential()
    if is_doubled_layer:
        model.add(LSTM(neurons, input_shape=x_shape, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons/2), input_shape=x_shape, return_sequences=True))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(neurons, input_shape=x_shape))
        model.add(Dropout(dropout_rate))
    if is_regularized:
        model.add(Dense(x_shape[1], 'relu',  kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(Dense(x_shape[1], 'relu'))
    model.add(Dense(y_shape, 'linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def visialize_loss_plot(history):
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.xlabel('# epochs')
    plt.ylabel('MSE')