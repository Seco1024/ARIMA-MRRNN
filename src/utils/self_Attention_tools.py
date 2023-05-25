import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.losses import MeanSquaredError
from keras import  metrics
from keras.optimizers import Adam
from sklearn.metrics import r2_score
import os
import re


parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

class self_Attention(object):
    def __init__(self, look_back, n_features, dropout, lr, num_transformer_blocks=4, head_size=64, num_heads=4, ff_dim=4):
        self.look_back = look_back
        self.n_features = n_features
        self.horizon = 1
        self.lr = lr

        self.num_transformer_blocks=num_transformer_blocks
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.mlp_units = [128]
        self.mlp_dropout = 0.4
        self.dropout = dropout

    def transformer_encoder(self, inputs):
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

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

        outputs = layers.Dense(self.horizon)(x)
        return keras.Model(inputs, outputs)

    def train(self, train_X, train_y, val_X, val_y, epochs, batch_size, today):
        self.model = self.build()
        adam = Adam(learning_rate=self.lr)
        mse = MeanSquaredError()
        self.model.compile(optimizer=adam, loss = mse, metrics=[metrics.MSE, metrics.MAE])

        early_stopping = EarlyStopping(patience=50, restore_best_weights=True)
        checkpoint_dir = os.path.join(parent_dir, f'models/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.h5')
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=10, min_lr=pow(0.7, 5)*float(self.lr))
        checkpoint = ModelCheckpoint(checkpoint_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=1,
                            callbacks=[early_stopping, checkpoint, lr_reduction])
        return self.history

    def restore(self, today):
        self.best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.h5'))
        adam = Adam(learning_rate=self.lr)
        mse = MeanSquaredError()
        self.best_model.compile(optimizer=adam, loss = mse, metrics=[metrics.MSE, metrics.MAE])
    
    def predict(self, residual_X):
        self_attention_prediction = self.best_model.predict(residual_X)
        return self_attention_prediction      
            
    def evaluate_model(self, train_X, train_y, val_X, val_y, test_X, test_y, today):
        self.best_model = load_model(os.path.join(parent_dir, f'models/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.h5'))
        score_train = self.best_model.evaluate(train_X, train_y)
        score_test = self.best_model.evaluate(test_X, test_y)
        score_val = self.best_model.evaluate(val_X, val_y)
        evaluate_df = pd.DataFrame(np.array([score_train[1], score_val[1], score_test[1], 
                                    score_train[2], score_val[2], score_test[2]]).reshape(-1, 6),
                        columns=["train_MSE", "val_MSE", "test_MSE", "train_MAE", "val_MAE", "test_MAE"])
        evaluate_df.to_csv(os.path.join(parent_dir, f'out/self_Attention_error/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.csv'), index=False)
        
    def visualize_loss_plot(self, today):
        plt.figure()
        plt.plot(self.history.history['mean_squared_error'], label='training MSE')
        plt.plot(self.history.history['val_mean_squared_error'], label='validation MSE')
        plt.legend()
        plt.xlabel('# epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(parent_dir, f'out/self_Attention_plot/{str(today)}/self_Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.png'))
        
    def visualize_prediction_plot(self, hybrid_prediction, original, arima_prediction, timestamps, today, file):
        plt.figure()
        ticker1, ticker2 = re.findall(r"\d+", str(file))[0], re.findall(r"\d+", str(file))[1]
        plt.plot(timestamps, hybrid_prediction, label= f'ARIMA-Self Attention Prediction Close')
        plt.plot(timestamps, original, label='Original Close')
        plt.plot(timestamps, arima_prediction, label='ARIMA Prediction Close')
        plt.legend()
        plt.xlabel('year')
        plt.ylabel('Correlation Coefficient')
        plt.title(f"ARIMA-Self Attention Prediction on {ticker1}-{ticker2}({today})")
        plt.savefig(os.path.join(parent_dir, f'out/hybrid_model_plot/{str(today)}/Self Attention_{self.num_transformer_blocks}_{self.head_size}_{self.num_heads}_{self.ff_dim}_{self.dropout}_{self.lr}.png'))