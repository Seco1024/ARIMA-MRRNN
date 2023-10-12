import numpy as np
import pandas as pd
import argparse
import logging
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from utils import rnn_tools
from utils.mrrnn_tools import (
    MutuallyRecursiveRNN,
    regularization_loss,
    list_to_tensor,
    shuffle,
)
from clearml import Task

parser = argparse.ArgumentParser()
parser.add_argument("--cell", default="lstm")
parser.add_argument("--epochs", default=300)
parser.add_argument("--batch", default=64)
parser.add_argument("--neurons", default=32, help="number of neurons")
parser.add_argument("--double_layer", default=0, help="is double layered")
parser.add_argument("--dropout", default=0.5, help="Dropout Rate")
parser.add_argument("--lr", default=0.001, help="learning rate")
parser.add_argument("--lookback", default=14)
# parser.add_argument('--double_layer', default=0, help="is double layered")
args = parser.parse_args()

task = Task.init(
    project_name="ARIMA-MRRNN",
    task_name=f"ARIMA-MRRNN({args.cell}) , lookback={str(args.lookback)}",
)
logging.basicConfig(level=logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

epochs = int(args.epochs)
batch = int(args.batch)
neurons = int(args.neurons)
dropout = float(args.dropout)
lr = float(args.lr)

today = rnn_tools.get_today()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
files_dir = os.path.join(parent_dir, f"data/VARMA_ARIMA/after_ARIMA_full/")
# files_dir = os.path.join(parent_dir, f"data/preprocessed_data_full/")
# if not os.path.exists(os.path.join(parent_dir, f"models/{str(today)}")):
#     os.makedirs(os.path.join(parent_dir, f"models/{str(today)}"))
#     os.makedirs(os.path.join(parent_dir, f"out/MRRNN_error/{str(today)}"))
#     os.makedirs(os.path.join(parent_dir, f"out/MRRNN_plot/{str(today)}"))
#     os.makedirs(os.path.join(parent_dir, f"out/hybrid_model_error/{str(today)}"))
#     os.makedirs(os.path.join(parent_dir, f"out/hybrid_model_plot/{str(today)}"))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ================== preprocessing phase
data = []
d = 0
for file in os.listdir(files_dir):
    file_path = os.path.join(files_dir, file)
    df = pd.read_csv(file_path, index_col="date", parse_dates=True)
    if not d:
        dates = df.index
        d = 1
    if np.isnan(df).any().any() or np.isnan(df).any().any():
        continue
    data.append(df.reset_index(drop=True))

X, y, train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], [], [], []
lookback = int(args.lookback)
future_n = 1

for pair_corr in data:
    # scaler.fit(pair_corr)
    # pair_corr = scaler.transform(pair_corr)
    for i in range(lookback, len(pair_corr) - future_n + 1):
        X.append(pair_corr[i - lookback : i])
        y.append(pair_corr[i + future_n - 1 : i + future_n]["close"])

X = np.array([X[i].values.tolist() for i in range(len(X))])
y = np.array([a.values.tolist() for a in y])
train_X, train_y = X[: int(len(X) * 0.7)], y[: int(len(X) * 0.7)]
val_X, val_y = (
    X[int(len(X) * 0.7) : int(len(X) * (0.7 + 0.15))],
    y[int(len(X) * 0.7) : int(len(X) * (0.7 + 0.15))],
)
test_X, test_y = X[int(len(X) * (0.7 + 0.15)) :], y[int(len(X) * (0.7 + 0.15)) :]
input_shape = (train_X.shape[1], train_X.shape[2])

# ================== training phase
mrrnn = MutuallyRecursiveRNN(
    input_shape[1],
    1,
    input_shape[1] - 1,
    int(int(args.neurons) / input_shape[1]),
    int(int(args.neurons) * (input_shape[1] - 1) / input_shape[1]),
    float(args.dropout),
    cell=args.cell,
    double_layer=True if int(args.double_layer) == 1 else False,
)

adam_optimizer = optim.Adam(mrrnn.parameters(), lr=float(lr))
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()
lr_scheduler = ReduceLROnPlateau(adam_optimizer, mode="min", factor=0.5, patience=10)
best_loss = np.inf
early_stopping = EarlyStopping(patience=16, 
                               verbose=True, 
                               path=os.path.join(parent_dir,f"models/{str(today)}/MRRNN_{args.cell}_{str(neurons)}_{str(dropout)}.pt")
                               )

training_loss_list, valid_loss_list = [], []

for epoch in range(int(epochs)):
    (
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    ) = ([], [], [], [], [], [])
    train_X, train_y, val_X, val_y, test_X, test_y = shuffle(
        train_X, train_y, val_X, val_y, test_X, test_y
    )

    for i in range(int(len(train_X) / batch)):
        train_X_tensor.append(train_X[i * batch : i * batch + batch])
        train_y_tensor.append(train_y[i * batch : i * batch + batch])

    for i in range(int(len(val_X) / batch)):
        val_X_tensor.append(val_X[i * batch : i * batch + batch])
        val_y_tensor.append(val_y[i * batch : i * batch + batch])
        test_X_tensor.append(test_X[i * batch : i * batch + batch])
        test_y_tensor.append(test_y[i * batch : i * batch + batch])

    (
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    ) = list_to_tensor(
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    )

    mrrnn.train()
    training_loss = 0.0
    for i in range(int(len(train_X) / batch)):
        x, y = train_X_tensor[i], train_y_tensor[i]
        adam_optimizer.zero_grad()
        price_pred, metrics_pred = mrrnn(x)
        loss = mse_criterion(price_pred, y)
        loss.backward()
        adam_optimizer.step()
        training_loss += loss.item()
    training_loss /= int(len(train_X) / batch)

    mrrnn.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for i in range(int(len(val_X) / batch)):
            x, y = val_X_tensor[i], val_y_tensor[i]
            price_pred, metrics_pred = mrrnn(x)
            loss = mse_criterion(price_pred, y)
            valid_loss += loss.item()

        valid_loss /= int(len(val_X) / batch)
        lr_scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                mrrnn.state_dict(),
                os.path.join(
                    parent_dir,
                    f"models/{str(today)}/MRRNN_{args.cell}_{str(neurons)}_{str(dropout)}.pt",
                ),
            )

        print(
            f"Epoch {epoch+1}: training loss: {training_loss}, Val Loss: {valid_loss}"
        )        
    early_stopping(valid_loss, mrrnn)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    training_loss_list.append(training_loss)
    valid_loss_list.append(valid_loss)

mrrnn.visualize_loss_plot(training_loss_list, valid_loss_list, today)
mrrnn.load_state_dict(
    torch.load(
        os.path.join(
            parent_dir,
            f"models/{str(today)}/MRRNN_{args.cell}_{args.neurons}_{args.dropout}.pt",
        )
    )
)

# ================== validating phase
mrrnn.eval()
with torch.no_grad():
    (
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    ) = ([], [], [], [], [], [])
    train_X, train_y, val_X, val_y, test_X, test_y = shuffle(
        train_X, train_y, val_X, val_y, test_X, test_y
    )
    train_mse, train_mae, val_mse, val_mae, test_mse, test_mae = 0, 0, 0, 0, 0, 0

    for i in range(int(len(train_X) / batch)):
        train_X_tensor.append(train_X[i * batch : i * batch + batch])
        train_y_tensor.append(train_y[i * batch : i * batch + batch])

    for i in range(int(len(val_X) / batch)):
        val_X_tensor.append(val_X[i * batch : i * batch + batch])
        val_y_tensor.append(val_y[i * batch : i * batch + batch])
        test_X_tensor.append(test_X[i * batch : i * batch + batch])
        test_y_tensor.append(test_y[i * batch : i * batch + batch])

    (
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    ) = list_to_tensor(
        train_X_tensor,
        train_y_tensor,
        val_X_tensor,
        val_y_tensor,
        test_X_tensor,
        test_y_tensor,
    )

    for i in range(int(len(train_X) / batch)):
        x, y = train_X_tensor[i], train_y_tensor[i]
        price_pred, metrics_pred = mrrnn(x)
        train_mse += mse_criterion(price_pred, y).item()
        train_mae += mae_criterion(price_pred, y).item()

    for i in range(int(len(test_X) / batch)):
        x, y = test_X_tensor[i], test_y_tensor[i]
        price_pred, metrics_pred = mrrnn(x)
        test_mse += mse_criterion(price_pred, y).item()
        test_mae += mae_criterion(price_pred, y).item()

        x, y = val_X_tensor[i], val_y_tensor[i]
        price_pred, metrics_pred = mrrnn(x)
        val_mse += mse_criterion(price_pred, y).item()
        val_mae += mae_criterion(price_pred, y).item()

    val_mse /= int(len(val_X) / batch)
    val_mae /= int(len(val_X) / batch)
    test_mse /= int(len(test_X) / batch)
    test_mae /= int(len(test_X) / batch)
    train_mse /= int(len(train_X) / batch)
    train_mae /= int(len(train_X) / batch)

evaluate_df = pd.DataFrame(
    np.array([train_mse, val_mse, test_mse, train_mae, val_mae, test_mae]).reshape(
        -1, 6
    ),
    columns=["train_MSE", "val_MSE", "test_MSE", "train_MAE", "val_MAE", "test_MAE"],
)
evaluate_df.to_csv(
    os.path.join(
        parent_dir,
        f"out/MRRNN_error/{str(today)}/MRRNN_{args.cell}_{str(args.neurons)}_{str(args.dropout)}.csv",
    ),
    index=False,
)
