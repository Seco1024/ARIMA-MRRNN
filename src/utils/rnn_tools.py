import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras import regularizers, metrics
import os
import datetime
import re


parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

def get_today():
    today = datetime.date.today()
    date_obj = datetime.datetime.strptime(f"{today}", "%Y-%m-%d")
    formatted_date = date_obj.strftime("%m%d")
    return formatted_date

def mean(target):
    mean_value = sum(target)/len(target)
    return mean_value

class rnn_model(object):
    def __init__(self, is_doubled_layer, neurons, dropout, lr, cell='LSTM'):
        self.is_doubled_layer = is_doubled_layer
        self.neurons = neurons
        self.dropout = dropout
        self.lr = lr
        self.cell = cell

    def build(self, x_shape, y_shape):
        self.model = Sequential()
        if self.is_doubled_layer and self.cell == 'LSTM':
            self.model.add(LSTM(self.neurons, input_shape=x_shape, return_sequences=True))
            self.model.add(Dropout(self.dropout))
            self.model.add(LSTM(self.neurons, input_shape=x_shape, return_sequences=True))
            self.model.add(Dropout(self.dropout))
        elif self.cell == 'LSTM':
            self.model.add(LSTM(self.neurons, input_shape=x_shape))
            self.model.add(Dropout(self.dropout))
        elif self.is_doubled_layer and self.cell == 'GRU':
            self.model.add(GRU(self.neurons, input_shape=x_shape, return_sequences=True))
            self.model.add(Dropout(self.dropout))
            self.model.add(GRU(self.neurons, input_shape=x_shape, return_sequences=True))
            self.model.add(Dropout(self.dropout))
        elif self.cell == 'GRU':
            self.model.add(GRU(self.neurons, input_shape=x_shape))
            self.model.add(Dropout(self.dropout))
        elif self.cell == 'RNN':
            self.model.add(SimpleRNN(self.neurons, input_shape=x_shape))
            self.model.add(Dropout(self.dropout))
            
        self.model.add(Dense(y_shape, 'linear', kernel_regularizer=regularizers.l2(1e-4)))
        
        adam = Adam(learning_rate=self.lr)
        mse = MeanSquaredError()
        self.model.compile(optimizer=adam, loss=mse, metrics=[metrics.MSE, metrics.MAE])
        return self.model
    
    def restore(self, today):
        self.best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/{self.cell}_{str(self.is_doubled_layer)}_{str(self.neurons)}_{str(self.dropout)}_{str(self.lr)}.h5'))
        adam = Adam(learning_rate=self.lr)
        mse = MeanSquaredError()
        self.best_model.compile(optimizer=adam, loss=mse, metrics=[metrics.MSE, metrics.MAE])
        
    def predict(self, residual_X):
        rnn_prediction = self.best_model.predict(residual_X)
        return rnn_prediction

    def visualize_loss_plot(self, history, today):
        plt.figure()
        plt.plot(history.history['mean_squared_error'], label='training MSE')
        plt.plot(history.history['val_mean_squared_error'], label='validation MSE')
        plt.legend()
        plt.xlabel('# epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(parent_dir, f'out/{self.cell}_plot/{str(today)}/{self.cell}_{str(self.is_doubled_layer)}_{str(self.neurons)}_{str(self.dropout)}_{str(self.lr)}.png'))
    
    def evaluate_model(self, train_X, train_y, val_X, val_y, test_X, test_y, today):
        best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/{self.cell}_{str(self.is_doubled_layer)}_{str(self.neurons)}_{str(self.dropout)}_{str(self.lr)}.h5'))
        score_train = best_model.evaluate(train_X, train_y)
        score_test = best_model.evaluate(test_X, test_y)
        score_val = best_model.evaluate(val_X, val_y)
        evaluate_df = pd.DataFrame(np.array([score_train[1], score_val[1], score_test[1], 
                                    score_train[2], score_val[2], score_test[2]]).reshape(-1, 6),
                        columns=["train_MSE", "val_MSE", "test_MSE", "train_MAE", "val_MAE", "test_MAE"])
        print(evaluate_df)
        evaluate_df.to_csv(os.path.join(parent_dir, f'out/{self.cell}_error/{str(today)}/{self.cell}_{str(self.is_doubled_layer)}_{str(self.neurons)}_{str(self.dropout)}_{str(self.lr)}.csv'), index=False)
    
    def visualize_prediction_plot(self, hybrid_prediction, original, arima_prediction, timestamps, today, file):
        plt.figure()
        ticker1, ticker2 = re.findall(r"\d+", str(file))[0], re.findall(r"\d+", str(file))[1]
        plt.plot(timestamps, hybrid_prediction, label= f'ARIMA-{self.cell} Prediction Close')
        plt.plot(timestamps, original, label='Original Close')
        plt.plot(timestamps, arima_prediction, label='ARIMA Prediction Close')
        plt.legend()
        plt.xlabel('year')
        plt.ylabel('Correlation Coefficient')
        plt.title(f"ARIMA-{self.cell} Prediction on {ticker1}-{ticker2}(neurons={self.neurons}, {today})")
        plt.savefig(os.path.join(parent_dir, f'out/hybrid_model_plot/{str(today)}/ARIMA-{self.cell}_{str(self.is_doubled_layer)}_{str(self.neurons)}_{str(self.dropout)}_{str(self.lr)}_({ticker1}-{ticker2}).png'))