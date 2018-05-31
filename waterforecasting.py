import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt #to show, plt.show()
import tensorflow as tf
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from data_in import data_in

data = data_in()

borough1 = data['BRONX']

### Attribution:
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    cols= [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)
    df = pd.concat(cols, axis=1)
    df.fillna(0, inplace=True)
    df.columns = ["Month", "Consumption (HCF)", "Next Month",
    "Consumption (HCF)"]
    return df

timeseries_to_supervised(borough1)
