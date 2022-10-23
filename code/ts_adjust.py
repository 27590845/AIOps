# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:59:28 2021

@author: D
"""

import pandas as pd
import numpy as np
import time
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(r'E:\My source\02   Project\2   Anormaly Detection and Root Cause Location\2   AIOps Challenge\AIOPs\CPU_error\venv\call_chain\\') #这里preprocess是split_by_date.py所在文件夹


import ts_features as tf


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


path = r'E:\My source\02   Project\2   Anormaly Detection and Root Cause Location\2   AIOps Challenge\AIOps挑战赛数据\2020_04_21\调用链指标\\'
trace_csf = pd.read_csv(path + 'trace_csf.csv')
trace_fly_remote = pd.read_csv(path + 'trace_fly_remote.csv')
trace_jdbc = pd.read_csv(path + 'trace_jdbc.csv')
trace_local = pd.read_csv(path + 'trace_local.csv')
trace_osb = pd.read_csv(path + 'trace_osb.csv')
trace_remote_process = pd.read_csv(path + 'trace_remote_process.csv')

# concat all
all_trace = pd.concat([trace_csf, trace_fly_remote, trace_jdbc, \
                       trace_local, trace_osb, trace_remote_process], axis=0, sort=False)
all_trace.reset_index(drop=True, inplace=True)

## time type
#all_trace["startTime"] = all_trace["startTime"].apply(lambda d: datetime.datetime.fromtimestamp(int(d) / 1000) \
#                                                      .strftime('%Y-%m-%d %H:%M:%S'))
#all_trace["startTime"] = pd.to_datetime(all_trace["startTime"])

# 预处理, CSF的预处理放在后面子集，省时间
all_trace.insert(all_trace.shape[1], 'newServiceName', np.NAN)
all_trace.loc[(all_trace.callType == 'JDBC'), 'newServiceName'] = all_trace.loc[
    (all_trace.callType == 'JDBC'), 'dsName']
all_trace.loc[(all_trace.callType == 'LOCAL'), 'newServiceName'] = all_trace.loc[
    (all_trace.callType == 'LOCAL'), 'dsName']
all_trace.loc[(all_trace.callType == 'RemoteProcess'), 'newServiceName'] = all_trace.loc[
    (all_trace.callType == 'RemoteProcess'), 'serviceName']

# 2 fault path
fault = pd.read_excel(
    r'E:\My source\02   Project\2   Anormaly Detection and Root Cause Location\2   AIOps Challenge\fault.xlsx')
#fault['start_time'] = pd.to_datetime(fault['start_time'])  # 标准数据

# 3 调用链时序分析
# 参数anomaly_id表示duration：取异常点前后2min30s 的数据分析, traceId 1000+,  取异常点前后1min 的数据分析, traceId 500+
anomaly_id = 13  # para1:fault的index
fault.index = fault['index']
start_time = fault.at[anomaly_id, 'start_time']
print("anomaly start at ", start_time)
start_time = datetime_timestamp(str(start_time))
#start_time = pd.to_datetime(start_time)
duration = 2*60*1000
all_trace_duration = all_trace.loc[(all_trace['startTime'] >= start_time - duration) &
                                   (all_trace['startTime'] < start_time + duration), :]


# CSF特殊处理
def fill_newServiceName(id_para, col):
    try:
        return all_trace_duration.loc[(all_trace_duration.pid == id_para), col].values[0]
    except:
        return np.NAN


# 对于CSF类的需要结合子调用，使用子调用的cmbdid作为本次调用的serviceName， all_trace_duration小点，跑快点
all_trace_CSF = all_trace_duration.loc[(all_trace_duration.callType == 'CSF'), :]
all_trace_CSF['newServiceName'] = all_trace_CSF.apply(lambda row: fill_newServiceName(row['id'], 'cmdb_id'), axis=1)

# 调用链个数统计
traceIds = all_trace_duration['traceId'].value_counts().to_frame()

# 参数 all_trace_duration的前 n_chain 个调用链, 分析主要调用链省时间
n_chain = 100  # para2 调用链数目

count = 0
for traceId in traceIds.head(n_chain).index:
    count = count+1
    print(count)
    oneCall = all_trace_duration.loc[all_trace_duration['traceId'] == traceId]
    for unq_id in oneCall['id'].unique():
        # 父耗时要减去所有子的耗时
        try:
            unq_id_child_elapsedTime = oneCall[oneCall['pid'] == unq_id]['elapsedTime'].sum()
        except:
            unq_id_child_elapsedTime = 0

        all_trace_duration.loc[all_trace_duration['id'] == unq_id, 'elapsedTime'] -= unq_id_child_elapsedTime

# 父亲的调用情况拼接到一行，方便分析数据
top_traceIds = traceIds.head(n_chain).index.to_list()
# 对n_chain个调用链补充 pid对应的 cmdb_id	serviceName	dsName	newServiceName
all_trace_duration.insert(all_trace_duration.shape[1], 'pid_cmdb_id', np.NAN)
all_trace_duration.insert(all_trace_duration.shape[1], 'pid_serviceName', np.NAN)
all_trace_duration.insert(all_trace_duration.shape[1], 'pid_dsName', np.NAN)
all_trace_duration.insert(all_trace_duration.shape[1], 'pid_newServiceName', np.NAN)

# 先只对 top_traceIds 的调用链做处理： ！！！
all_trace_duration_top_traceIds = all_trace_duration.loc[all_trace_duration['traceId'].isin(top_traceIds)]


def func_pid(pid, col):
    try:
        return all_trace_duration_top_traceIds.loc[all_trace_duration_top_traceIds['id'] == pid, col].values[0]
    except:
        return np.NAN


all_trace_duration_top_traceIds['pid_newServiceName'] = \
    all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'], 'newServiceName'), axis=1)

all_trace_duration_top_traceIds['pid_cmdb_id'] = \
    all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'], 'cmdb_id'), axis=1)

all_trace_duration_top_traceIds['pid_serviceName'] = \
    all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'], 'serviceName'), axis=1)

all_trace_duration_top_traceIds['pid_dsName'] = \
    all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'], 'dsName'), axis=1)

# 计算异常分数矩阵
# 双层遍历1： cmdb_id	serviceName	dsName 的时序图看看情况
pid_col_name = 'pid_cmdb_id'  # 'pid_newServiceName'
id_col_name = 'cmdb_id'  # 'newServiceName'


interval_point = 5

# 每行是pid， 列对应这 id，被调用的
pid_cmdb_names = all_trace_duration_top_traceIds[pid_col_name].unique()
id_cmdb_names = all_trace_duration_top_traceIds[id_col_name].unique()

df_scores = pd.DataFrame(index=pd.Index(pid_cmdb_names.tolist()),
                         columns=pd.Index(id_cmdb_names.tolist()))
array_series = np.array(df_scores)

dict_pid = dict(zip(range(len(pid_cmdb_names)), pid_cmdb_names))
dict_pid_rev = dict(zip(pid_cmdb_names, range(len(pid_cmdb_names))))

dict_id = dict(zip(range(len(id_cmdb_names)), id_cmdb_names))
dict_id_rev = dict(zip(id_cmdb_names, range(len(id_cmdb_names))))

for pid_cmdb in pid_cmdb_names:
    for id_cmdb in id_cmdb_names:

        # 取出 col_name 里面取值为 cmdb 的调用
        all_trace_duration_top_traceIds_cmdb = \
            all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds[pid_col_name] == pid_cmdb) & \
                                                (all_trace_duration_top_traceIds[id_col_name] == id_cmdb)]

        if (all_trace_duration_top_traceIds_cmdb.shape[0] < 2 * interval_point):
            continue;

        # 按时间排序
        all_trace_duration_top_traceIds_cmdb.sort_values(by=['startTime'], ascending=True, inplace=True)

        pid = dict_pid_rev.get(pid_cmdb)
        id = dict_id_rev.get(id_cmdb)
        array_series[pid, id] = all_trace_duration_top_traceIds_cmdb


window_size = 1*1000
interval = 1*1000
start_time
duration
# 规整后的start_time
new_start_time = [start_time-duration]
for i in range(int(2*duration/interval)-1):
    new_start_time.append(new_start_time[i]+interval)


def ts_adjust(df_series, window_size, new_start_time):
    #规整后的elapsed_time
    new_elapsed_time  = np.zeros(len(new_start_time))
    
    for i in range(len(new_elapsed_time)):
        df_temp = df_series.loc[(df_series['startTime'] >= new_start_time[i]-window_size) &
                                          (df_series['startTime'] < new_start_time[i]+window_size), :]
        if len(df_temp) != 0:
            # 时间窗内有数据，取所有数据的平均
            array_elapsed_time = np.array(df_temp['elapsedTime'])
            new_elapsed_time[i] = np.mean(array_elapsed_time)
        else:
            #如果时间窗内没数据，取最近时间点的数据
            array_start_time = np.array(df_series['startTime'])
            array_elapsed_time = np.array(df_series['elapsedTime'])
            new_elapsed_time[i] = array_elapsed_time[np.argmin(np.abs(array_start_time-new_start_time[i]))]
            
    new_df_series = pd.DataFrame(columns = ['startTime', 'elapsedTime'])
    new_df_series['startTime'] = new_start_time
    new_df_series['elapsedTime'] = new_elapsed_time
    
    return new_df_series




# 矩阵形式的规整时序
new_array_series = np.array(df_scores)
for j in range(array_series.shape[0]):
    for k in range(array_series.shape[1]):
        df_series = array_series[j,k]
        if type(df_series) != float:
           new_df_series = ts_adjust(df_series, window_size, new_start_time) 
           new_array_series[j,k] = new_df_series
           
# list形式的规整时序       
new_series = []#[[j-cmdb,k-cmdb,时序],...]格式，j,k行列可由字典查对应的id,pid容器
for j in range(new_array_series.shape[0]):
    for k in range(array_series.shape[1]):
        temp = new_array_series[j,k]
        if type(temp) != float:
            new_series.append([dict_pid.get(j),dict_id.get(k),temp])


# 单条时间序列特征提取
single_ts_features = [] 
for i in range(len(new_series)):
    temp = new_series[i][2]
    cmdb = [new_series[i][:2]]
    a1 = tf.basic_info(temp)
    a2 = tf.variation_start_time(temp)
    a3 = tf.kde(temp, a2[0])
    single_ts_features.append(cmdb+a1+a2+a3)
    
df_single_ts_features = pd.DataFrame(single_ts_features)
df_single_ts_features.columns=['cmdb', 'max', 'min', 'median', 'mean',
                'std','variation_start_time', 'gradient', 'po', 'pu']
            
   
# 两条时间序列特征提取
double_ts_features = [] 
for i in range(len(new_series)):
    tempi = new_series[i][2] 
    cmdbi = [new_series[i][:2]]
    for j in range(i+1, len(new_series)):
        tempj = new_series[j][2]
        cmdbj = [new_series[j][:2]]
        cmdb = [cmdbi+cmdbj]
        d1 = tf.pearson(tempi, tempj)
        d2 = tf.granger(tempi, tempj)
        d3 = tf.granger(tempj, tempi)
        double_ts_features.append(cmdb+d1+d2+d3)     

df_double_ts_features = pd.DataFrame(double_ts_features)
# granger21=1:后面的cmdb是前面cmdb的因
df_double_ts_features.columns = ['cmdb', 'pearson', 'granger21', 'granger12']