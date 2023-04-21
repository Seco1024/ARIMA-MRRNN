import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import argparse


def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC') # .dropna() handles differenced data
    if result[1] <= 0.05:    # Data has no unit root and is stationary
        return True
    else:      # Data has a unit root and is non-stationary
        return False
