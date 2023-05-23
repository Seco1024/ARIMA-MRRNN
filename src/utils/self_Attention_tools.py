import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from time import time
import json
import logging
import os

import tensorflow as tf
import keras
from keras import layers
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import MeanSquaredError
from keras import regularizers, metrics
from sklearn.metrics import r2_score

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

class self_Attention(object):
    def __init__(self, look_back, n_features, horizon, num_transformer_blocks=4, head_size=64, num_heads=4, ff_dim=4):
        self.look_back = look_back
        self.n_features = n_features
        self.horizon = horizon

        self.head_size=head_size
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.num_transformer_blocks=num_transformer_blocks
        self.mlp_units=[128]
        self.mlp_dropout=0.4
        self.dropout=0.25


    def transformer_encoder(self, inputs):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)
        
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


    def build(self):
        inputs = keras.Input(shape=(self.look_back, self.n_features))
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        # output layer
        outputs = layers.Dense(self.horizon)(x)
        return keras.Model(inputs, outputs)


    def restore(self, filepath):
        self.best_model = load_model(filepath)
        self.best_model.compile(optimizer='adam', loss = ['mse'], metrics=[metrics.MSE, metrics.MAE])


    def train(self, X_train, y_train, today, epochs=100, batch_size=64):
        """ Training the network
        :param X_train: training feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_train: training target vectors
        :type 2-D Numpy array of float values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """

        self.model = self.build()
        mse = MeanSquaredError()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = mse, metrics=[metrics.MSE, metrics.MAE])

        early_stopping = EarlyStopping(patience=50, restore_best_weights=True)
        checkpoint_dir = os.path.join(parent_dir, f'models/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}.h5')
        checkpoint = ModelCheckpoint(checkpoint_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0001)

        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping, checkpoint, lr_reduction])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])
        return self.history


    def evaluate(self, X_test, y_test):
        """ Evaluating the network
        :param X_test: test feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_test: test target vectors
        :type 2-D Numpy array of int values
        :return  Evaluation losses
        :rtype 5 Float tuple
        :raise -
        """

        y_pred = self.model.predict(X_test)

        # Print accuracy if ground truth is provided
        """
        if y_test is not None:
            loss_ = session.run(
                self.loss,
                feed_dict=feed_dict)
        """
        _, mse_result, mae_result, _ = self.model.evaluate(X_test, y_test)
        # r2_result = r2_score(y_test.flatten(),y_pred.flatten())
        return _, mse_result, mae_result
    
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
            
            
    def evaluate_model(self, model, train_X, train_y, val_X, val_y, test_X, test_y, today):
        score_train = model.evaluate(train_X, train_y)
        score_test = model.evaluate(test_X, test_y)
        score_val = model.evaluate(val_X, val_y)
        evaluate_df = pd.DataFrame(np.array([score_train[1], score_val[1], score_test[1], 
                                    score_train[2], score_val[2], score_test[2]]).reshape(-1, 6),
                        columns=["train_MSE", "val_MSE", "test_MSE", "train_MAE", "val_MAE", "test_MAE"])
        evaluate_df.to_csv(os.path.join(parent_dir, f'out/self_Attention_error/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}.csv'), index=False)
        
        
    def visualize_loss_plot(self, today):
        plt.figure()
        plt.plot(self.history.history['mean_squared_error'], label='training MSE')
        plt.plot(self.history.history['val_mean_squared_error'], label='validation MSE')
        plt.legend()
        plt.xlabel('# epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(parent_dir, f'out/self_Attention_plot/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}.png'))