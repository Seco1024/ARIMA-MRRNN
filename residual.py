import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

residual_path='./data/VARMA_ARIMA/residual'
file_n=0
for root, dir, files in os.walk(residual_path):
    for file in files:
        exec(f"residual_df{file_n} = pd.read_csv(os.path.join(root, file))")
        file_n+=1
residuals = [eval(f"residual_df{i}") for i in range(file_n)]
