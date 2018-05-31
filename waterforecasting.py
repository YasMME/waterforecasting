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
print(data)

### Attribution:
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def timeseries_to_supervised(data, lag=1):
    #empty
    return null
