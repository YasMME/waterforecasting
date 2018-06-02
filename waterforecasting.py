import numpy as np
import pandas as pd
from data_in import data_in
import matplotlib.pyplot as plt #to show, plt.show()
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense
from keras.models import Sequential
from math import sqrt

data = data_in()
borough1 = data['BRONX']

### Attribution:
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

X = borough1.values
raw_vals = pd.DataFrame(X)

#create supervised learning problem
def series_to_sv(series, lag=1):
    df = pd.DataFrame(series)
    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)
    df = pd.concat(cols, axis=1)
    df.fillna(0, inplace=True)
    return df

def get_trend(series):
    trend = series.diff()
    trend = trend.fillna(0)[1]
    return trend 

#returns consumption value given history, difference value, 
#and interval
def untrend(history, diff, interval=1):
    val = history.iloc[-interval-1]
    return diff + val[1]

def split_data(series):
    l = len(series)
    split = (2*l)/3
    train = series[0:split]
    test = series[split+1:l]
    return train, test 

def scale_series(train, test):
    x = train.values
    x = np.array(x)
    x = x.reshape(len(x), 2)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(x)
    scaled_train = scaler.transform(x)
    scaled_train = pd.DataFrame(scaled_train)
    scaled_test = scaler.transform(test.values)
    scaled_test = pd.DataFrame(scaled_test)
    return scaler, scaled_train, scaled_test

def unscale(scaler, X, value): 
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverse = scaler.inverse_transform(array)
    return inverse[0, -1]

### Attr. Machine Learning Mastery
def fit_lstm(train, batch_size, num_epochs, neurons):
    x, y = np.array(train.iloc[:,0:1]), np.array(train.iloc[:,1:2])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1],
        x.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    for i in range(num_epochs):
        model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0,
                shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    forecast = model.predict(X, batch_size=batch_size)
    return forecast[0,0]

trend_data = get_trend(raw_vals)
supervised_vals = series_to_sv(trend_data)
train, test = split_data(supervised_vals)
scaler, scaled_train, scaled_test = scale_series(train, test)

lstm_model=fit_lstm(scaled_train, 1, 200, 3)
train_reshaped = np.array(scaled_train.iloc[:,0]).reshape(len(scaled_train), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

predictions = list()
for i in range(len(scaled_test)):
    X, y = scaled_test.iloc[i, 0:-1], scaled_test.iloc[i, -1]
#    forecast = forecast_lstm(lstm_model, 1, X)
    forecast = y
    forecast = unscale(scaler, X, forecast)
    forecast = untrend(raw_vals, forecast, len(scaled_test)+1-i)
    predictions.append(forecast)
    expected = raw_vals.iloc[len(train) + i + 1][1]
    print("Month=%d, Predicted=%f, Expected=%f" % (i+1, forecast,
        expected))


rmse = sqrt(mean_squared_error(raw_vals.iloc[-len(test):][1], predictions))
print('Test RMSE: %.3f' % rmse)
plt.plot(raw_vals.iloc[-len(test):][1])
plt.plot(predictions)
plt.show()
