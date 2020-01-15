# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:27:12 2020

@author: ns_10
"""

import os
import pandas as pd
import random
import matplotlib.pyplot as plt

data_path = r'.\Dataset\Stocks'
# data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
filenames = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.txt') and os.path.getsize(os.path.join(data_path,f)) > 0]

data = []
plt.figure()
for file in filenames[:10]:
    
    df = pd.read_csv(file)
    
    df['Label'] = file.split('\\')[-1].split('.')[0]
    df['Date'] = pd.to_datetime(df['Date'])
    data.append(df)
    
    #plot visualization
    df['Close'].plot(style='--',label=df['Label'][0])
    
plt.title('Sample of Stock Data')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()