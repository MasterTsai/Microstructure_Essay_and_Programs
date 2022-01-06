# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:28:15 2019

@author: Cai Xiao 翻版必究
"""

import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import statsmodels as sm #statsmodels用不了，statsmodels.api可以
import numpy as np

#plt.rcParams['figure.figsize'] = (8.0, 4.0) # 设置figure_size尺寸
#plt.rcParams['savefig.dpi'] = 2000 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
#plt.savefig(‘plot123_2.png’, dpi=200)#指定分辨率
my_path = "D:\\HNU\\Market Microstructure\\YiLi" #股票地址
save_path = "D:\\HNU\\Market Microstructure" #计算结果保存本地文件夹
files_name = os.listdir(my_path) #文件名加后缀
#读取日期
fdates2017 = []
for fname in files_name:
    fdates, fextension = os.path.splitext(fname)   
    fdates2017.append(fdates)

# 读取文件
prices_all_year = pd.Series() #创建空的Series
ln_returns_all_year = pd.Series()
rv_day = pd.Series() #每日的日内已实现波动率

# Order-Flow
orderflow = []
for i in files_name:
    files_path = os.path.join(my_path, i)    
    data = pd.read_excel(files_path, index_col=False) #有表头，列无index
    buy = data.iloc[:, 14]
    sell = data.iloc[:, 15]
    buy_sum = sum(buy.values) 
    sell_sum = sum(sell.values)
    orderflow_meitian = buy_sum - sell_sum
    orderflow.append(orderflow_meitian)
np.save(save_path + 'orderflow.npy', orderflow)
    
# 每天有几个对数收益率数据 （价格数据 - 1）
M_eachday = []
for j in files_name:
    files_path = os.path.join(my_path, j)
    data = pd.read_excel(files_path, index_col=False)
    M_temp = data.shape[0] - 1
    M_eachday.append(M_temp)
np.save('D:\\HNU\\Market Microstructure\\每日对数收益率个数.npy', M_eachday)

# 保存收益率文件 
save_path_return_tick = 'D:\\HNU\\Market Microstructure\\YiLi_returns'
for i in files_name:
    files_path = os.path.join(my_path, i)    
    data = pd.read_excel(files_path, index_col=False) #有表头，列无index
    prices = data.iloc[:, 1]
    ln_prices = pd.Series([math.log(x) for x in prices])
    temp = ln_prices.diff()[1:]
    save_path_tick_data = os.path.join('D:\\HNU\\Market Microstructure\\YiLi_returns', i[:10])
    np.save(save_path_tick_data, temp)

# Question 4 计算考虑噪音和跳跃时的已实现波动率 (tick data)
for i in files_name:
    files_path = os.path.join(my_path, i)    
    data = pd.read_excel(files_path, index_col=False) #有表头，列无index
    prices = data.iloc[:, 1]
    ln_prices = pd.Series([math.log(x) for x in prices])
    temp = ln_prices.diff()[1:]
    a = 0    
    for x in temp.values:
        a += x**2
    rv = pd.Series([a])
    #print(sm.tsa.stattools.adfuller(ln_returns.diff()[1:]))
    prices_all_year = prices_all_year.append(prices)
    rv_day = rv_day.append(rv)
rv_day.index = fdates2017
#print(rv_day)
#plt.figure(dpi=144)
#plt.grid(linestyle='-.')
#plt.plot(range(1,240), rv_day, 'p-.')
#month = ['2017-01','2017-02','2017-03','2017-04','2017-05','2017-06'
#         ,'2017-07','2017-08','2017-09','2017-10','2017-11','2017-12']
#month_index = [0, 18, 36, 59, 73, 92, 114, 135, 158, 179, 196, 218]
#plt.xticks(month_index, month, rotation=0)

# Question 1 计算RBV (tick data)

rbv_day = pd.Series()

for j in files_name:
    files_path = os.path.join(my_path, j)    
    data = pd.read_excel(files_path, index_col=False) #有表头，列无index
    prices = data.iloc[:, 1]
    ln_prices = pd.Series([math.log(x) for x in prices])
    temp = ln_prices.diff()[1:]
    b = 0
    M = temp.values.shape[0] #一天内有多少个收益率数据
    for x in range(2, M):
        b += abs(temp.values[x]) * abs(temp.values[x-2])
    b = b * (math.pi/2) * (M/(M-2)) 
    print(b)    
    rbv = pd.Series([b])
    #print(sm.tsa.stattools.adfuller(ln_returns.diff()[1:]))
    rbv_day = rbv_day.append(rbv)
rbv_day.index = fdates2017

#保存文件
np.save('D:\\HNU\\Market Microstructure\\rv_day_tick.npy', rv_day)
np.save('D:\\HNU\\Market Microstructure\\rbv_day_tick.npy', rbv_day)
np.save('D:\\HNU\\Market Microstructure\\fdates2017.npy', fdates2017)
np.save('D:\\HNU\\Market Microstructure\\files_name.npy', files_name)

# Question 2 5min data
fivemin_per = 5 / (24 * 60)
am_start = 9.5 / 24
am_end = 11.5 / 24
pm_start = 13.0 / 24
pm_end = 15.0 / 24

rv_5min = []

for i in files_name:

    save_path_5min = "D:\\HNU\\Market Microstructure\\YiLi_5min"
    
    files_path = os.path.join(my_path, i)
    data = pd.read_excel(files_path, index_col=False)
    tick_time = data.iloc[:, 0]
    prices = data.iloc[:, 1]
    M = tick_time.values.shape[0] #看看与多少个价格（或时间）数据
    ii = 0
    prices_5min = list()
    for j in range(0, 24):
        prices_5_temp = 0    
        ii_begin = ii    
        
        while (round(tick_time.values[ii],6) >= round(am_start+j*fivemin_per,6)) and (round(tick_time.values[ii],6) <= round(am_start+(j+1)*fivemin_per,6)):
            prices_5_temp += prices.values[ii]
            ii += 1
            ii_end = ii - 1
        temp_num = ii_end - ii_begin + 1
        if temp_num:    
            prices_5_avg = prices_5_temp / temp_num    
            prices_5min.append(prices_5_avg)
        else:
            prices_5min.append(prices_5min[j-1])
    
    j_num = len(prices_5min) #上午有多少个五分钟价格数据
    
    for j in range(0, 24):
        prices_5_temp = 0    
        ii_begin = ii    
        
        while (round(tick_time.values[ii],6) >= round(pm_start+j*fivemin_per,6)) and (round(tick_time.values[ii],6) <= round(pm_start+(j+1)*fivemin_per,6)):
            if ii == M-1:
                ii_end = ii
                prices_5_temp += prices.values[ii]            
                break          
            prices_5_temp += prices.values[ii]
            ii += 1
            ii_end = ii - 1        
                  
    
        temp_num = ii_end - ii_begin + 1
        if temp_num:    
            prices_5_avg = prices_5_temp / temp_num    
            if prices_5_avg:
                prices_5min.append(prices_5_avg)
        else:
            prices_5min.append(prices_5min[j+j_num-1])
    
    #计算 5min_rv
    ln_prices_5min = []
    num_5min = 0
    returns_5min = []
    rerurns_5min_square = []
    
    ln_prices_5min = [math.log(prices) for prices in prices_5min ]
    num_5min = len(ln_prices_5min)    
    returns_5min = [ln_prices_5min[j+1] - ln_prices_5min[j] for j in range(0, num_5min-1)]   
    rerurns_5min_square = [x**2 for x in returns_5min]
    rv_5min.append(sum(rerurns_5min_square))
    
    save_5min = os.path.join(save_path_5min, i[:10])
    np.save(save_5min, prices_5min)            

rv_5min00 = {'rv_5min':rv_5min}
rv_5min_df = pd.DataFrame(rv_5min00)
rv_5min_df.index = fdates2017
np.save('D:\\HNU\\Market Microstructure\\rv_5min.npy', rv_5min_df)

# 示性函数 indicator function
#Calculate Z-stat
# Calculate C
# Calculate J

rv_day = np.load('D:\\HNU\\Market Microstructure\\rv_day_tick.npy')
rbv_day = np.load('D:\\HNU\\Market Microstructure\\rbv_day_tick.npy')

JUMP = []
CONTI = []

alpha = 0.001 
nppf = 3.090232306
 #显著性水平为0.999，求分位点

#示性函数
def ind_dayu(x):
    if x > nppf:
        i = 1
    else:
        i = 0
    return i
def ind_xiaoyu(x):
    if x <= nppf:
        i = 1
    else:
        i = 0
    return i
for t in range(0, 239):
    # Calculate rtq
    #跳跃部分
    jump = 0  
    #连续部分
    conti = 0
    npy_name = fdates2017[t] + '.npy'
    npv_r_tick_path = os.path.join(save_path_return_tick, npy_name)
    temp_return = np.load(npv_r_tick_path)
    
    rtq_sigma = 0    
    M_temp = M_eachday[t]
    constant_part = M_temp * (M_temp / (M_temp - 4)) * ((2**(2/3)) * (math.gamma(7/6) / math.gamma(1/2)))**(-3)
    
    for j in range(4, M_temp):
        rtq_sigma += abs(temp_return[j-4])**(4/3) * abs(temp_return[j-2])**(4/3) * abs(temp_return[j])**(4/3)
    
    rtq = constant_part * rtq_sigma
    # Calculate Z-stat
    z_stat = M_temp ** (1/2) * ((rv_day[t]-rbv_day[t])/rv_day[t])/((((math.pi/2)**2+math.pi-5)*max(1,(rtq/rbv_day[t]**2)))**(1/2))
    jump = ind_dayu(z_stat) * (rv_day[t] - rbv_day[t])
    JUMP.append(jump)
    conti = ind_dayu(z_stat) * rbv_day[t] + ind_xiaoyu(z_stat) * rv_day[t]
    CONTI.append(conti)

np.save('D:\\HNU\\Market Microstructure\\JUMP.npy', JUMP)
np.save('D:\\HNU\\Market Microstructure\\CONTI.npy', CONTI)
jt = np.load('D:\\HNU\\Market Microstructure\\JUMP.npy')  
ct = np.load('D:\\HNU\\Market Microstructure\\CONTI.npy')

# 描述性统计 最大值 最小值 平均值 中位数 标准差
def basic_stats(x):
    max00 = np.max(x)
    min00 = np.min(x)
    avg00 = np.mean(x)
    mid00 = np.median(x)
    std00 = np.std(x)
    return max00, min00, avg00, mid00, std00
    
plt.figure(dpi=200)
plt.plot(range(1,240),jt,'r--', linewidth=1.5)
plt.title('Jump & Continuous')
plt.plot(range(1,240),ct,'b-',  linewidth=1.5)
plt.plot(range(1,240),rv_day,'g:',  linewidth=1.5)
plt.legend(['jump', 'continuous', 'rv']) # 图例
plt.xlabel('Date')
plt.ylabel('')
plt.show()

#跳跃部分占比
ptg = []
for i in range(0, 239):
    ptg00 = jt[i] / rv_day[i]
    
    ptg.append(100*ptg00)
plt.plot(range(1,240),ptg,'g:',  linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.title('Jump/RV')
plt.show()

x = orderflow
y = np.array([14,24,18,17,27])

# 回归方程求取函数
def fit(x,y):
    if len(x) != len(y):
        return
    numerator = 0.0
    denominator = 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(len(x)):
        numerator += (x[i]-x_mean)*(y[i]-y_mean)
        denominator += np.square((x[i]-x_mean))
    print('numerator:',numerator,'denominator:',denominator)
    b0 = numerator/denominator
    b1 = y_mean - b0*x_mean
    return b0,b1

# 定义预测函数
def predit(x,b0,b1):
    return b0*x + b1

# 求取回归方程
b0,b1 = fit(x,y)
print('Line is:y = %2.0fx + %2.0f'%(b0,b1))

# 预测
x_test = np.array([0.5,1.5,2.5,3,4])
y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test[0][i] = predit(x_test[i],b0,b1)

# 绘制图像
xx = np.linspace(0, 5)
yy = b0*xx + b1
plt.plot(xx,yy,'k-')
plt.scatter(x,y,cmap=plt.cm.Paired)
plt.scatter(x_test,y_test[0],cmap=plt.cm.Paired)
plt.show()
