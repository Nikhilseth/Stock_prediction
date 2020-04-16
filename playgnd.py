#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:23:52 2020

@author: gregoireg
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__) 

#%%
stock_names = ["aapl", "ibm", "amd", "intc", "msft"]

for i in range(len(stock_names)):
    stock_names[i] +=".us.txt"

data_path = r'\.\Dataset\Stocks'
print(os.listdir(data_path))
#%%
#data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
filenames = [os.path.join(data_path, f) for f in stock_names]

data = []
for file in filenames:
    
    df = pd.read_csv(file)
    
    df['Label'] = file.split('\\')[-1].split('.')[0]
    df['Date'] = pd.to_datetime(df['Date'])
    data.append(df)

## Windows
split_time = 121
window_size = 10
batch_size = 32
predict_size = 1 #The model fails if you make this >1, doesn't seem to be able to handle it
close_train = data[0]['Close'][:-split_time] #, data[0]['Open'][-20:], data[0]['High'][-20:], data[0]['Low'][-20:]]
#print(close_train.size)
close_test = data[0]['Close'][-split_time:]
#print(close_train.shape)
def create_ds(ds):
    dataset = tf.expand_dims(ds, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size + predict_size, shift=window_size + predict_size, drop_remainder= True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size + predict_size))
    dataset = dataset.map(lambda x: (x[:-predict_size], x[-predict_size:]))
#    dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def predict_ds(train_set):
    ds = tf.expand_dims(train_set, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.window(window_size, shift=window_size + predict_size, drop_remainder= True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1) 
    forecast = model.predict(ds)
    return forecast

dataset = create_ds(close_train)

print(dataset)
#for x, y in dataset:
#    print(x.numpy())
#    print(y.numpy())
    
#%% Model Creation & Fit
#    data1 = data[0].drop(['Open','High','Low','Date','Volume','Label','OpenInt'],1)
#    data2 = data[0]['Close']
#    print(data1)
#    print(tf.expand_dims(data2, axis=-1))

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(dataset, epochs=50)


#%% Prediction

forecast = predict_ds(close_test)
forecast = forecast[split_time - window_size:-1, -1, 0]
#print(dataset)
#print(close_test)
print(forecast.shape)
#print(ds)
#for x in ds:
#    print(x)

#forecast = model.predict(close_test[0:window_size])
#print(close_test)
#print(forecast)

#plt.figure()
#plt.plot(close_test,label='actual')
#plt.plot(forecast[0],label='predicted')
#plt.legend()
#plt.title('Predicted and true outputs from LSTM Model: ' + df['Label'][0])
#plt.ylabel('Closing Price')
#plt.xlabel('Time')

#%%
#tf.keras.metrics.mean_absolute_error(close_test, forecast).numpy()
