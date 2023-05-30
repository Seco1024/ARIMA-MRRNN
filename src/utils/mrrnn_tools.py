import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
import random
import datetime
import re

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

class MutuallyRecursiveRNN(nn.Module):
    def __init__(self, features_dim, target_dim, explanatory_dim, target_embedding_dim, explanatory_embedding_dim, dropout_rate):
        super(MutuallyRecursiveRNN, self).__init__()
        
        self.price_embedding_dim = target_embedding_dim
        self.metrics_embedding_dim = explanatory_embedding_dim
        self.features_dim = features_dim
        self.price_dim = target_dim
        self.metrics_dim = explanatory_dim     
        self.dropout = dropout_rate 
        
        self.price_input_dim = self.metrics_embedding_dim + self.features_dim
        self.metrics_input_dim = self.price_embedding_dim + self.features_dim
        
        self.price_rnn = nn.RNNCell(self.price_input_dim, self.price_embedding_dim)
        self.metrics_rnn = nn.RNNCell(self.metrics_input_dim, self.metrics_embedding_dim)
        
        self.price_linear_layer = nn.Linear(self.price_embedding_dim, self.price_dim)
        self.metrics_linear_layer = nn.Linear(self.metrics_embedding_dim, self.metrics_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        
    def forward(self, input_seq):
        batch_size = len(input_seq)
        seq_length = len(input_seq[0])
        self.price_embedding = torch.zeros(batch_size, self.price_embedding_dim)
        self.metrics_embedding = torch.zeros(batch_size, self.metrics_embedding_dim)
        
        for i in range(seq_length):
            self.metrics_embedding = self.metrics_rnn(torch.cat((input_seq[:, i, :], self.price_embedding), dim=1), self.metrics_embedding)
            self.price_embedding = self.price_rnn(torch.cat((input_seq[:, i, :], self.metrics_embedding), dim=1), self.price_embedding)
        
        metrics_pred = self.metrics_linear_layer(self.dropout_layer(self.metrics_embedding))
        price_pred = self.price_linear_layer(self.dropout_layer(self.price_embedding))
        
        return price_pred, metrics_pred
    
    def visualize_loss_plot(self, training_loss, valid_loss, today):
        plt.figure()
        plt.plot(training_loss, label='training MSE')
        plt.plot(valid_loss, label='validation MSE')
        plt.legend()
        plt.xlabel('# epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(parent_dir, f'out/MRRNN_plot/{str(today)}/MRRNN_{str(self.price_embedding_dim * self.features_dim)}_{str(self.dropout)}.png'))

def regularization_loss(params, mode='l2', lambda_reg=0.001):
    l1_reg = torch.tensor(0., requires_grad=True)
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in params:
        l2_reg += torch.sum(torch.pow(param, 2))
        l1_reg += torch.sum(torch.abs(param))
    if mode == 'l2':
        loss = lambda_reg * l2_reg
    elif mode == 'l1':
        loss = lambda_reg * l1_reg
    else:
        raise ValueError("Undefined regularization mode, choose either 'l1' or 'l2'")
    return loss

def shuffle(train_X, train_y, val_X, val_y, test_X, test_y):
    train_pair = list(zip(train_X, train_y))
    random.shuffle(train_pair)
    val_pair = list(zip(val_X, val_y))
    random.shuffle(val_pair)
    test_pair = list(zip(test_X, test_y))
    random.shuffle(test_pair)
    
    train_X, train_y = zip(*train_pair)
    val_X, val_y = zip(*val_pair)
    test_X, test_y = zip(*test_pair)
    
    return train_X, train_y, val_X, val_y, test_X, test_y


def list_to_tensor(train_X_tensor, train_y_tensor, val_X_tensor, val_y_tensor, test_X_tensor, test_y_tensor):
    train_X = torch.Tensor(train_X_tensor)
    train_y = torch.Tensor(train_y_tensor)
    val_X = torch.Tensor(val_X_tensor)
    val_y = torch.Tensor(val_y_tensor)
    test_X = torch.Tensor(test_X_tensor)
    test_y = torch.Tensor(test_y_tensor)
    return train_X, train_y, val_X, val_y, test_X, test_y