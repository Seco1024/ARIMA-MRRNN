import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import regularizers, metrics
from sklearn.preprocessing import MinMaxScaler
from lib import lstm_tools
import os
import datetime

def get_today():
    today = datetime.date.today()
    date_obj = datetime.datetime.strptime(today, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%m%d")
    return formatted_date

parent_dir = parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
today = get_today()

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
    
    adam = Adam(lr=0.001)
    mse = MeanSquaredError()
    model.compile(optimizer=adam, loss=mse, metrics=[metrics.MSE, metrics.MAE])
    return model

def visialize_loss_plot(history):
    plt.plot(history.history['mean_squared_error'], label='training MSE')
    plt.plot(history.history['val_mean_squared_error'], label='validation MSE')
    plt.legend()
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(parent_dir, f'out/LSTM_plot/{today}.png'))
    
def evaluate_model(model, train_X, train_y, val_X, val_y, test_X, test_y):
    score_train = model.evaluate(train_X, train_y)
    score_test = model.evaluate(test_X, test_y)
    score_val = model.evaluate(val_X, val_y)
    evaluate_df = pd.DataFrame(np.array([score_train[1], score_val[1], score_test[1], 
                                score_train[2], score_val[2], score_test[2]]).reshape(-1, 6),
                      columns=["train_MSE", "val_MSE", "test_MSE", "train_MAE", "val_MAE", "test_MAE"])
    evaluate_df.to_csv(os.path.join(parent_dir, f'out/LSTM_error/{today}.csv'), index=False)