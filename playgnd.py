#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:23:52 2020

@author: gregoireg
"""

import os
 
stock_names = ["aapl", "ibm", "amd", "hpq", "xrx", "msft"]

for i in range(len(stock_names)):
    stock_names[i] +=".us.txt"

data_path = os.path.expanduser("~/Documents/Github/Dataset/Stocks") #for Mac
filenames1 = [os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.txt') and os.path.getsize(os.path.join(data_path,f)) > 0]
filenames2 = [os.path.join(data_path, f) for f in stock_names]
    
print(filenames1[:5])
print(filenames2[:5])