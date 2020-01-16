# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:09:42 2020

@author: ns_10
"""

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout

def build_model(inputs,output_size,neurons,activ='linear',dropout=0.1,loss='mae',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons,input_shape=(inputs.shape[1],inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size,activation=activ))
    
    model.compile(loss=loss,optimizer=optimizer)
    return model