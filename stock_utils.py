# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:09:42 2020

@author: ns_10
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

def build_model(inputs,output_size,neurons,activ='linear',dropout=0.1,loss='mae',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons,input_shape=(inputs.shape[1],inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size,activation=activ))
    
    model.compile(loss=loss,optimizer=optimizer)
    return model

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def load_data(filenames):
    data = []
    for file in filenames:
        df = pd.read_csv(file)
        df['Label'] = file.split('\\')[-1].split('.')[0]
        df['Date'] = pd.to_datetime(df['Date'])
        data.append(df)
    return data

def plot_sample_data(data):
    r = lambda: np.random.randint(0,255)
    traces = []
    for df in data:
        clr_str = 'rgb('+str(r())+','+str(r())+','+str(r())+')'
        df = df.sort_values('Date')
        label = df['Label'].iloc[0]
        trace = go.Scattergl(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color=clr_str),
            name=label)
        traces.append(trace)
    layout = go.Layout(title='Sample plot of Stocks')
    fig = go.Figure(data=traces,layout=layout)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Close Price')
    fig.show()
    
def plot_predictions(predictions,true_output,title=''):
    #plot predictions
    plt.figure()
    plt.plot(true_output,label='actual')
    plt.plot(predictions,label='predicted')
    plt.legend()
    plt.title(title)
    plt.ylabel('Closing Price')
    plt.xlabel('Time')
    