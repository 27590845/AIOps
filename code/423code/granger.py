# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:02:06 2021

@author: CZY
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from minepy import MINE

# Unix时间转换为日期时间
def timestamp_datetime(value):
    """
    作用：Unix时间戳转换成标准时间
    参数：
    value: Unix时间戳（毫秒级别）
    返回：
    dt: 标准时间结构："%Y-%m-%d %H:%M:%S"
    """
    form = '%Y-%m-%d %H:%M:%S'
    # value为传入的值为时间戳(整形)（毫秒），如：1332888820*1000
    value = value / 1000
    value = time.localtime(value)
    # # 经过localtime转换后变成 # time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40,
    # tm_wday=2, tm_yday=88, tm_isdst=0) 最后再经过strftime函数转换为正常日期格式。
    dt = time.strftime(form, value)
    return dt


# 日期时间转换为Unix时间
def datetime_timestamp(dt):
    """
    作用：标准时间转换成Unix时间戳
    参数：
    dt: 标准时间结构："%Y-%m-%d %H:%M:%S"
    返回：
    value: Unix时间戳（毫秒级别）
    """
    # dt为字符串
    # 中间过程，一般都需要将字符串转化为时间数组
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    # # time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88,
    # tm_isdst=-1) 将"2012-03-28 06:53:40"转化为时间戳
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s) * 1000

# 取一段时间窗的指标：type:'itemid'/'cmdb_name'，iid:itemid的int/[cmdb_id,name]
def find_timeseries_interval(df, iid, timestart=None, timeend=None, coltype="itemid"):
    """
    功能：根据给定的dataframe, iid, timestart, timeend, 从dataframe中找出<=timeend 且>=timestart的所有timestamp和对应的value

    参数：
    df: dataframe,至少包括列:itemid, timestamp, value
    iid:当type=='itemid'时，为需要寻找的itemid值，int类型(对于str类型代码中做了int转换)；
    当flag=='cmdb_name'时，为需要寻找的cmdb_id+name值，
        list类型，里面两个str，格式为[cmdb_id, name]
    timestart: 时间开始点，需要找>=该时间点的timestamp, 若为None，则从序列开头取
    timeend: 时间开始点，需要找<=该时间点的timestamp, 若为None，则一直取到序列结尾
    type: 只能输入两个值：'itemid'或'cmdb_name'，以此确定iid的格式
    返回：
    timeseries: 对应iid的所有timestamp和值， 属于 [timestart,timeend]，且按照升序排列，df类型
    """
    # 判断type书写是否正确
    if coltype == 'itemid':
        iid = int(iid)
        ts = df.loc[df['itemid'] == iid]
    elif coltype == 'cmdb_name':
        try:
            iid[0]
        except:
            print("iid值应为一个列表")
            return None
        df1 = df.loc[df['cmdb_id'] == iid[0]]
        ts = df1.loc[df1['name'] == iid[1]]
    else:
        print("type类型错误")
        return None

    # 按照输入timestart，timeend格式提取df
    if timestart is None and timeend is not None:
        ts_time = ts.loc[ts['timestamp'] <= timeend]
    elif timestart is not None and timeend is None:
        ts_time = ts.loc[ts['timestamp'] >= timestart]
    elif timestart is not None and timeend is not None:
        ts_time = ts.loc[ts['timestamp'] <= timeend]
        ts_time = ts_time.loc[ts_time['timestamp'] >= timestart]
    else:
        ts_time = ts

    # 排序，提取时间序列
    ts_sort = ts_time.sort_values(by=['timestamp'])
    timeseries = ts_sort[['timestamp', 'value']]

    return timeseries


def print_stats(mine):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    a = mine.mic()
    return a

def granger(metric1, metric2):
    """
    Parameters
    ----------
    metric1 : TYPE: DataFrame
        DESCRIPTION. 只包含指标1的value的dataframe
    metric2 : TYPE: DataFrame
        DESCRIPTION. 只包含指标2的value的dataframe

    
    Returns type: bool
        DESCRIPTION. 1 or 0, metric2 是否为 metric1 的因果关系

    """
    df = pd.concat([metric1 , metric2] , axis=1)
    maxlag = 10
    result = grangercausalitytests(df,maxlag,verbose=False)
    min_pvalue = 1
    
    for k in range(1,maxlag+1):
        #取出 lag=k 时的pvalue
        temp_value = result.get(k)[0]['ssr_ftest'][1] 
        if temp_value < min_pvalue:
            min_pvalue = temp_value
        
    if min_pvalue <= 0.05:
        return 1
    else:
        return 0

def same_gap_adjust(df_1, df_2):
    """
    Parameters
    ----------
    df_1 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value
    df_2 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value

    Returns
    -------
    None.

    """

    df_1 = df_1.reset_index(drop = True)
    df_2 = df_2.reset_index(drop = True)
    
    interval = df_1['timestamp'][1]-df_1['timestamp'][0]
    
    #df = [df_1, df_2]
    timestamp_1 = list(df_1['timestamp'])
    timestamp_2 = list(df_2['timestamp'])
    
    later_start_time = min(timestamp_1[0],timestamp_2[0])
    first_end_time = min(timestamp_1[-1],timestamp_2[-1])
    
    df_1_cut = df_1[df_1['timestamp']>= later_start_time]
    df_1_cut = df_1_cut[df_1_cut['timestamp']<= first_end_time]
    df_1_cut = df_1_cut.reset_index(drop = True)
    
    df_2_cut = df_2[df_2['timestamp']>= later_start_time]
    df_2_cut = df_2_cut[df_2_cut['timestamp']<= first_end_time]
    df_2_cut = df_2_cut.reset_index(drop = True)
    
    #df_cut = [df_1_cut, df_2_cut]
    
    cut_1_stime = df_1_cut['timestamp'][0]
    cut_2_stime = df_2_cut['timestamp'][0]
   
    gap = abs(cut_1_stime - cut_2_stime)
    
    if gap > 0.5*interval:
        if cut_1_stime > cut_2_stime:
            df_1_cut = df_1_cut.drop(index=0)
            df_1_cut = df_1_cut.reset_index(drop = True)
            len_1 = len(df_1_cut)
            len_2 = len(df_2_cut)
            if len_2 > len_1:
                df_2_cut = df_2_cut.drop(index=len(df_2_cut)-1)
        else:
            df_2_cut = df_2_cut.drop(index=0)
            df_2_cut = df_2_cut.reset_index(drop = True)
            len_2 = len(df_2_cut)
            len_1 = len(df_1_cut)
            if len_1 > len_2:
                df_1_cut = df_1_cut.drop(index=len(df_1_cut)-1)
    else:
         len_1 = len(df_1_cut)
         len_2 = len(df_2_cut)
         
         if len_2 > len_1:
             df_2_cut = df_2_cut.drop(index=len(df_2_cut)-1)
         elif len_1 > len_2:
             df_1_cut = df_1_cut.drop(index=len(df_1_cut)-1)
             
    return df_1_cut, df_2_cut
        
    
    





    
def unify_freq(df_1, df_2):
    """
    Parameters
    ----------
    df_1 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value
    df_2 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value

    Returns
    -------
    None.

    """
    df_1 = df_1.reset_index(drop = True)
    time_interval_1 = df_1['timestamp'][1]-df_1['timestamp'][0]
    
    df_2 = df_2.reset_index(drop = True)
    time_interval_2 = df_2['timestamp'][1]-df_2['timestamp'][0]
    
    #df = [df_1, df_2]
    #interval = [time_interval_1, time_interval_2]
    
    uni_interval = min (time_interval_1, time_interval_2)
    
    if time_interval_1 > time_interval_2:
        # 1.补齐采样频率大的序列
       
        m = round(time_interval_1/uni_interval) 
        
        timestamp_1 = list(df_1['timestamp'])
        timestamp_1_start = timestamp_1[0]
        timestamp_1_end = timestamp_1[-1]
        num_1 = round((timestamp_1_end - timestamp_1_start)/uni_interval)
        num_1 = int (num_1)
        num_1 = num_1 + 1
        new_timestamp_1 = list(np.linspace(timestamp_1_start, timestamp_1_end, num_1))
        
        value_1 = list(df_1['value'])
        new_value_1 =list(np.repeat(value_1,m))
        
        new_df_1 = pd.DataFrame({'timestamp':new_timestamp_1,'value':new_value_1})
        return new_df_1, df_2
    
    elif time_interval_2 > time_interval_1:
         # 1.补齐采样频率大的序列
        m = round(time_interval_2/uni_interval )
        
        timestamp_2 = list(df_2['timestamp'])
        timestamp_2_start = timestamp_2[0]
        timestamp_2_end = timestamp_2[-1]
        num_2 = round((timestamp_2_end - timestamp_2_start)/uni_interval)
        num_2 = int(num_2)
        num_2 =+ 1
        new_timestamp_2 = list(np.linspace(timestamp_2_start, timestamp_2_end, num_2))
        
        value_2 = list(df_2['value'])
        new_value_2 =list(np.repeat(value_2,m))
        
        new_df_2 = pd.DataFrame({'timestamp':new_timestamp_2,'value':new_value_2})
        return df_1, new_df_2
        
    else:
        return df_1, df_2
        
def unify_timestamp(df_1, df_2):
    """
    Parameters
    ----------
    df_1 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value
    df_2 : TYPE —— DataFrame
        DESCRIPTION. 两列，分别为timestamp和value

    Returns
    -------
    None.

    """
    df_1, df_2 = unify_freq(df_1, df_2)
    df_1, df_2 = same_gap_adjust(df_1, df_2)
    
    return df_1, df_2
   
    
    
    
    
    
if __name__ == '__main__':
    
    '''
    db_filedir_23 = r"E:\AiOps\AIOps_datasets_2020\2020_04_23\平台指标\db_oracle_11g.csv"  
    df_db_23 = pd.read_csv(db_filedir_23)   
    '''
   
    os_filedir = r"E:\AiOps\AIOps_datasets_2020\2020_04_23\平台指标\os_linux.csv"
    df_os = pd.read_csv(os_filedir)
    
    cmbd_id ='os_017'
    '''
    'os_017'
    df_Sq = find_timeseries_interval(df_os, [cmbd_id,'Sent_queue'],coltype="cmdb_name")
    df_Sq_value = df_Sq.reset_index(drop = True)['value']
    
    df_Rq = find_timeseries_interval(df_os, [cmbd_id,'CPU_util_pct'],coltype="cmdb_name")
    df_Rq_value = df_Rq.reset_index(drop = True)['value']
    
    df=pd.concat([df_Rq.reset_index(drop = True) , df_Sq.reset_index(drop = True)] , axis=1)
    #df_1 = pd.concat([df_Sq_value , df_Rq_value] , axis=1)
    #df_2 = pd.concat([df_Rq_value , df_Sq_value] , axis=1)
    '''
    
    df_1 = find_timeseries_interval(df_os, [cmbd_id,'Outgoing_network_traffic'],coltype="cmdb_name")
    df_2 = find_timeseries_interval(df_os, [cmbd_id,'Memory_available'],coltype="cmdb_name")
    
    df_1 = df_1.reset_index(drop = True)
    time_interval_1 = df_1['timestamp'][1]-df_1['timestamp'][0]
    
    df_2 = df_2.reset_index(drop = True)
    time_interval_2 = df_2['timestamp'][1]-df_2['timestamp'][0]
    
    #df = [df_1, df_2]
    #interval = [time_interval_1, time_interval_2]
    
    uni_interval = min (time_interval_1, time_interval_2)
    
    if time_interval_1 > time_interval_2:
        # 1.补齐采样频率大的序列
       
        m = round(time_interval_1/uni_interval )
        
        timestamp_1 = list(df_1['timestamp'])
        timestamp_1_start = timestamp_1[0]
        timestamp_1_end = timestamp_1[-1]
        num_1 = round((timestamp_1_end - timestamp_1_start)/uni_interval)
        num_1 = int (num_1)
        num_1 = num_1 + 1
        new_timestamp_1 = list(np.linspace(timestamp_1_start, timestamp_1_end, num_1,endpoint=True))
        
        value_1 = list(df_1['value'])
        new_value_1 =list(np.repeat(value_1,m))
        
        new_df_1 = pd.DataFrame({'timestamp':new_timestamp_1,'value':new_value_1})

    
    elif time_interval_2 > time_interval_1:
         # 1.补齐采样频率大的序列
        m = round(time_interval_2/uni_interval) 
        
        timestamp_2 = list(df_2['timestamp'])
        timestamp_2_start = timestamp_2[0]
        timestamp_2_end = timestamp_2[-1]
        num_2 = round((timestamp_2_end - timestamp_2_start)/uni_interval)
        num_2 = int(num_2)
        num_2 =+ 1
        new_timestamp_2 = list(np.linspace(timestamp_2_start, timestamp_2_end, num_2,endpoint=True))
        
        value_2 = list(df_2['value'])
        new_value_2 =list(np.repeat(value_2,m))
        
        new_df_2 = pd.DataFrame({'timestamp':new_timestamp_2,'value':new_value_2})
       
    
    
   # df_1, df_2 = unify_freq(df_Sq, df_Rq)
    
    
  
