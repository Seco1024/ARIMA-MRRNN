import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
parser = argparse.ArgumentParser()
parser.add_argument('--window', required=True, help='Enter windowsize')
args = parser.parse_args()

residual_path = os.path.join(parent_dir, f'out/VARMA_ARIMA_error/window{args.window}')
file_n = 0
for root, dir, files in os.walk(residual_path):
    for file in files:
        exec(f"residual_df{file_n} = pd.read_csv(os.path.join(root, file))")
        file_n+=1
residuals = [eval(f"residual_df{i}") for i in range(file_n)]
residuals_mean = pd.DataFrame(pd.concat(residuals).mean(numeric_only=True), columns=['MSE error'])
residuals_mean.to_csv(os.path.join(parent_dir, f'out/VARMA_ARIMA_error/window{args.window}/mean{args.window}.csv'), columns=['MSE error'])