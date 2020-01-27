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

print(tf.__version__)

stock_names = ["aapl", "ibm", "amd", "hpq", "xrx", "msft"]

for i in range(len(stock_names)):
    stock_names[i] +=".us.txt"

# data_path = r'.\Dataset\Stocks'
data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
filenames = [os.path.join(data_path, f) for f in stock_names]

data = []
for file in filenames:
    
    df = pd.read_csv(file)
    
    df['Label'] = file.split('\\')[-1].split('.')[0]
    df['Date'] = pd.to_datetime(df['Date'])
    data.append(df)

## Windows
window_size = 10
batch_sze = 2
close_open = data[0]['Close'][-20:] #, data[0]['Open'][-50:]]
# print(close_open)
dataset = tf.data.Dataset.from_tensor_slices(close_open)
dataset = dataset.window(window_size + 1, shift=1, drop_remainder= True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
# dataset = dataset.shuffle(10)
dataset = dataset.batch(batch_sze).prefetch(1)


# print(dataset)
for x, y in dataset:
    print(x.numpy())
    print(y.numpy())
