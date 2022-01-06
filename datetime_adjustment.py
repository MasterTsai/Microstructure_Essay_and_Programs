# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:48:53 2019

@author: CX
"""

import os
import datetime

# 时间戳
def get_timestamp(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()

#stamp转datetime 格式:list
def stamp2datetime(date_stamp):
    dt = [datetime.datetime.fromtimestamp(x) for x in date_stamp]
    return dt

#datatime转str 格式：list
def datetime2str(dt):
    date_str = [x.strftime("%Y-%m-%d") for x in dt]
    return date_str



my_path = "D:\\HNU\\Market Microstructure\\YiLi"

files_name = os.listdir(my_path)    
##切片处理
fdates2017 = []
for fname in files_name:

#原文件名 fdates
    fdates, fextension = os.path.splitext(fname)
    fdates = fdates[8:]   
    fdates2017.append(fdates)

fdates2017_stamp = [get_timestamp(x) for x in fdates2017]

fdates2017_stamp = sorted(fdates2017_stamp)
#时间戳转换成datatime格式
fdates2017 = stamp2datetime(fdates2017_stamp)
#datatime转成str
fdates2017_str = datetime2str(fdates2017) 

if __name__ == "__main__":
    #批量更改文件名
    for i in files_name:
        old_files = os.path.join(my_path, i)    
        fdates, fextension = os.path.splitext(i)
        fdates = fdates[8:]
        f_sta = get_timestamp(fdates)
        f_d = datetime.datetime.fromtimestamp(f_sta)
        f_d_str = f_d.strftime("%Y-%m-%d")
        new_files = os.path.join(my_path, f_d_str + '.xlsx')
        os.rename(old_files, new_files)
    

    
    
    
    