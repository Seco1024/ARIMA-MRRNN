import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

residual_path='./out/VARMA_ARIMA_residual'
file_n=0
for root, dir, files in os.walk(residual_path):
    for file in files:
        exec(f"residual_df{file_n} = pd.read_csv(os.path.join(root, file))")
        file_n+=1
residuals = [eval(f"residual_df{i}") for i in range(file_n)]
residuals_mean = pd.DataFrame(pd.concat(residuals).mean(numeric_only=True), columns=['RMSE error'])
residuals_mean.to_csv('./out/VARMA_ARIMA_residual/mean.csv', columns=['RMSE error'])