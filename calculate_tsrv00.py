# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:37:57 2020

@author: Cai Xiao
"""

import math
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

my_path = "D:\\HNU\\Market Microstructure\\YiLi"
save_path_ln_prices = "D:\\HNU\\Market Microstructure\\ln_prices_tick"
save_path_prices = "D:\\HNU\\Market Microstructure\\prices_tick"
files_name = os.listdir(my_path) #文件名加后缀
#读取日期
fdates2017 = []
for fname in files_name:
#原文件名 fdates
    fdates, fextension = os.path.splitext(fname)   
    fdates2017.append(fdates)

for i in files_name:
    files_path = os.path.join(my_path, i)    
    data = pd.read_excel(files_path, index_col=False) #有表头，列无index
    prices = data.iloc[:, 1]
    ln_prices = pd.Series([math.log(x) for x in prices])
    s = os.path.join(save_path_ln_prices, i[:10])
    s0 = os.path.join(save_path_prices, i[:10])
    np.save(s, ln_prices.values)
    np.save(s0, prices.values)

# 刘梦瑶 counteveryday:每日对数收益率个数
kk = 20
jj = 1
tsrv = []


m_eachday = np.load('D:\\HNU\\Market Microstructure\\每日对数收益率个数.npy')

for i in range(0, 239):
    # 导入对数价格    
    temp_names = os.path.join("D:\\HNU\\Market Microstructure\\ln_prices_tick"
                              , fdates2017[i] + '.npy')
    logprice = np.load(temp_names)    
    sparsekk = 0
    for j in range(0, m_eachday[i] + 1 - kk):
        sparsekk += (logprice[j+kk] - logprice[j]) ** 2
    average_sparsekk = sparsekk / kk
    
    lagjj = 0
    
    for j in range(0, m_eachday[i] + 1 - jj):
        lagjj += (logprice[j+jj] - logprice[j]) ** 2
    average_lagjj = lagjj / jj
    
    nbarkk = (m_eachday[i] - kk) / kk
    nbarjj = (m_eachday[i] - jj) / jj
    tsrv_meitian = (1-nbarkk/nbarjj)**(-1)*(average_sparsekk-nbarkk/nbarjj*average_lagjj)
    tsrv.append(tsrv_meitian)

np.save('D:\\HNU\\Market Microstructure\\tsrv.npy', tsrv)

rv_day = np.load('D:\\HNU\\Market Microstructure\\rv_day_tick.npy')
plt.figure(dpi=250)
plt.plot(range(1,240),tsrv,'ro-', label = 'TSRV', linewidth=1.5)
plt.title('TSRV')
plt.legend('TSRV')
plt.plot(range(1,240),rv_day,'b-', label = 'RV')
plt.legend(['TSRV', 'RV']) # 图例
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()