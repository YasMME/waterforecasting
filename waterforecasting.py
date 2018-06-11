import numpy as np
import pandas as pd
from data_in import data_in
import matplotlib.pyplot as plt #to show, plt.show()
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras import metrics
from math import sqrt

data = data_in()
borough1 = data['BRONX']

X = borough1.values
raw_vals = pd.DataFrame(X)


### Attribution: Adapted from
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
#create supervised learning problem
def series_to_sv(series, lag=1):
    df = pd.DataFrame(series)
    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)
    df = pd.concat(cols, axis=1)
    df.fillna(0, inplace=True)
    return df

def split_data(series):
    l = len(series)
    split = (2*l)/3
    train = series[0:split]
    test = series[split+1:l]
    return train, test 

### Attribution: Adapted from
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def scale_series(train, test):
    x = train.values[:, [1,3]]
    x = pd.DataFrame(x)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(x)
    scaled_train = scaler.transform(x)
    scaled_train = pd.DataFrame(scaled_train)
    y = test.values[:, [1,3]]
    y = pd.DataFrame(y)
    scaled_test = scaler.transform(y)
    scaled_test = pd.DataFrame(scaled_test)
    return scaler, scaled_train, scaled_test


### Attribution: Adapted from
###https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def unscale(scaler, X, value): 
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverse = scaler.inverse_transform(array)
    return inverse[0, -1]

def fit_lstm(train, batch_size, num_epochs, neurons):
    x, y = np.array(train.iloc[:,0:1]), np.array(train.iloc[:,1:2])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    for i in range(5):
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1],
            x.shape[2]), return_sequences=True, stateful=True))
        model.add(LSTM(neurons, return_sequences=True, stateful=True))
        model.add(LSTM(neurons))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(x, y, epochs=num_epochs, validation_split =
            0.33, batch_size=batch_size, verbose=2, shuffle=False)
        r = range(0, num_epochs)
        plt.plot(r, history.history['loss'], 'r', label="loss")
        plt.plot(r, history.history['val_loss'], 'c', label="val_loss")
        plt.legend(loc="upper right")
        plt.xlabel("epochs")
        plt.savefig(str(i)+"_"+str(num_epochs)+"loss.png")
        plt.clf()
        model.reset_states()
    return model, history

def forecast_lstm(model, batch_size, X):
    X = np.reshape(X, (1, 1, len(X)))
    forecast = model.predict(X, batch_size=batch_size)
    return forecast[0,0]

supervised_vals = series_to_sv(raw_vals)
train, test = split_data(supervised_vals)
scaler, scaled_train, scaled_test = scale_series(train, test)

lstm_model, history = fit_lstm(scaled_train, 1, 125, 32)
#train_reshaped = np.array(scaled_train.iloc[:,0]).reshape(len(scaled_train), 1, 1)
#lstm_model.predict(train_reshaped, batch_size=1)

predictions = list()
expected = list()
for i in range(len(scaled_test)):
    X, y = scaled_test.iloc[i, 0:-1], scaled_test.iloc[i, -1]
    forecast = forecast_lstm(lstm_model, 1, X)
    forecast = unscale(scaler, X, forecast)
    predictions.append(forecast)
    expect = raw_vals.iloc[len(train) + i + 1][1]
    expected.append(expect)
    print("Month=%d, Predicted=%f, Expected=%f" % (i+1, forecast,
        expect))

rmse = sqrt(mean_squared_error(expected, predictions))
#t = range(0, len(test))
print('Test RMSE: %.3f' % rmse)
#plt.plot(t, expected, t, predictions)
#plt.show()

