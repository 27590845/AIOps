# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:01:26 2021

@author: D
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

#特征提取
#Part I. 单条时间序列
#i.耗时：max, min, median, average, std
def basic_info(temp):
    """
    Parameters
    ----------
    temp: TYPE DataFrame
        DESCRIPTION. 时间序列的DataFrame，columns包含[startTime, elapsedTime]

    Returns
    -------
    包含耗时：max, min, median, average, std的array

    """
    info = [max(temp['elapsedTime']), 
            min(temp['elapsedTime']), 
            np.median(temp['elapsedTime']),
            np.mean(temp['elapsedTime']),
            np.std(temp['elapsedTime'])]
    
    return info
    
#ii.变化开始时间：对原时间序列进行一阶差分后，取梯度超过2.5 倍标准差的第一个时点作为变化开始时间。
#   若没有超过2.5 倍标准差的点，则取梯度绝对值排名前2 的两个点中时间在前的时间点作为变化开始时间。
def variation_start_time(ts_temp):
    """
    Parameters
    ----------
    ts_temp : TYPE DataFrame
        DESCRIPTION. 时间序列的DataFrame，columns包含[start_time, elapsedTime]

    Returns
    -------
    start_time:TYPE TimeStamp
        Description: 该时间序列的变化开始时间
    gradient:TYPE float
        Descrioption: 变化开始事件的变化梯度
    """  
    temp = ts_temp.sort_values(by = 'startTime')
    temp.insert(temp.shape[1], 'diff_1', np.NAN)
    temp['diff_1'] = temp.loc[:,'elapsedTime'].diff()
    temp.insert(temp.shape[1], 'diff_1_abs', np.NAN)
    temp['diff_1_abs'] = abs(temp.loc[:,'diff_1'])
   
    diff_1_std = float(np.nanstd(temp['diff_1']))
    n = 2.5
   
    diff_1_over_threshould = temp.loc[(temp['diff_1_abs'] > n*diff_1_std), :]
    if diff_1_over_threshould.empty:
        temp_sort = temp.sort_values(by=['diff_1_abs'],ascending=False)
        temp_sort = temp_sort.reset_index()
        start_time = temp_sort.loc[0, 'startTime']
        gradient = temp_sort.loc[0, 'diff_1_abs']
    
    else:
        start_time_index = diff_1_over_threshould.index[0]
        start_time = temp.loc[start_time_index, 'startTime']
        gradient = temp.loc[start_time_index, 'diff_1_abs']
    
    return [start_time, gradient]



def kde(ts_temp, change_start_time):
    """
    作用：用KDE方法计算某条时间序列change_start_time后overflow和underflow的概率
    参数：
    ts_temp: dataframe, 包含[startTime, elapsedTime]
    change_start_time: 变化开始时间，timestamp
    返回：
    apo: overflow概率
    pu: underflow概率
    """
    # temp标准化，[0,1]之间
    width = 0.1
    temp = ts_temp.sort_values(by = 'startTime')
    temp.insert(temp.shape[1], 'elapsed_std', np.NAN)
    X_min = np.min(temp['elapsedTime'])
    X_max = np.max(temp['elapsedTime'])
    X_std = (temp['elapsedTime'] - X_min) / (X_max - X_min)
    temp['elapsed_std'] = X_std
    
    X_train = np.array(temp[temp['startTime']<change_start_time]['elapsed_std'])
    X_test = np.array(temp[temp['startTime']>=change_start_time]['elapsed_std'])
    
    if len(X_train)<=2 or len(X_test)<=2:
        return [0, 0]
    else:
        train_value = X_train.reshape(-1, 1)
    
        # 建立KDE模型
        d = KernelDensity(kernel='gaussian', bandwidth=width).fit(train_value)
    
        # 计算概率值
        po = 0
        pu = 0
        for t in range(len(X_test)):
            # overflow 概率
            prob_o = quad(lambda x: np.exp(d.score_samples(np.array(x).reshape(-1, 1))), X_test[t], 5)[0]
            # underflow 概率
            prob_u = quad(lambda x: np.exp(d.score_samples(np.array(x).reshape(-1, 1))), -5, X_test[t])[0]
            po += np.log(prob_o)
            pu += np.log(prob_u)
        po = po / len(X_test)
        pu = pu / len(X_test)
    
        return [po, pu]
    
def pearson(temp1, temp2):
    """
    coefficient: 
        temp1:dataframe, 包含elapsedTime
        temp2:dataframe, 包含elapsedTime
        temp1,temp2等长
    return pearson_coef: 皮尔逊相关系数
    """
    
    return [np.corrcoef(temp1['elapsedTime'], temp2['elapsedTime'])[0,1]]


def granger(temp1, temp2):
    """
    Parameters
    ----------
    temp1 : TYPE: DataFrame
        DESCRIPTION. 包含elapsedTime
    temp2 : TYPE: DataFrame
        DESCRIPTION. 包含elapsedTime，和metric1等长

    Returns type: bool
        DESCRIPTION. 1 or 0, metric2 是否为 metric1 的因果关系

    """
    metric1 = temp1.sort_values(by = 'elapsedTime')['elapsedTime']
    metric2 = temp2.sort_values(by = 'elapsedTime')['elapsedTime']

    df = pd.concat([metric1, metric2], axis=1)
    maxlag = 10
    result = grangercausalitytests(df, maxlag,verbose=False)
    min_pvalue = 1

    for k in range(1, maxlag + 1):
        # 取出 lag=k 时的pvalue
        temp_value = result.get(k)[0]['ssr_ftest'][1]
        if temp_value < min_pvalue:
            min_pvalue = temp_value

    if min_pvalue <= 0.05:
        return [1]
    else:
        return [0]

def draw_ts(temp):
    plt.plot(temp['startTime'], temp['elapsedTime'])