import pandas as pd
import numpy as np

'''

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
'''

import datetime
import time
'''
from graphviz import Digraph
from IPython.display import display, Image
import os
import math
from wand.image import Image
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import requests
'''

import warnings
warnings.filterwarnings('ignore')

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

# path = '/Users/wangxue/Desktop/AIOps挑战赛/data_release_v2.0_20200411/调用链指标/'
path = r'E:\AiOps\AIOps_datasets_2020\2020_04_21\调用链指标\\'

# 1 读取调用链
trace_csf = pd.read_csv(path + 'trace_csf.csv')
trace_fly_remote = pd.read_csv(path +'trace_fly_remote.csv')
trace_jdbc = pd.read_csv(path + 'trace_jdbc.csv')
trace_local = pd.read_csv(path + 'trace_local.csv')
trace_osb = pd.read_csv(path +'trace_osb.csv')
trace_remote_process = pd.read_csv(path + 'trace_remote_process.csv')
# concat all
all_trace = pd.concat([trace_csf, trace_fly_remote, trace_jdbc, \
                  trace_local, trace_osb, trace_remote_process], axis=0, sort=False)


all_trace.reset_index(drop=True, inplace=True)
# time format
#all_trace["startTime"] = all_trace["startTime"].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000)\
#                                   .strftime('%Y-%m-%d %H:%M:%S'))
#all_trace["startTime"] = pd.to_datetime(all_trace["startTime"])
print("all trace read finish")
all_trace.head(5)

# 预处理, CSF的预处理放在后面子集，省时间
all_trace.insert(all_trace.shape[1], 'newServiceName', np.NAN)

all_trace.loc[(all_trace.callType == 'JDBC'),'newServiceName'] = all_trace.loc[(all_trace.callType == 'JDBC'),'dsName']
all_trace.loc[(all_trace.callType == 'LOCAL'),'newServiceName'] = all_trace.loc[(all_trace.callType == 'LOCAL'),'dsName']
all_trace.loc[(all_trace.callType == 'RemoteProcess'),'newServiceName'] = all_trace.loc[(all_trace.callType == 'RemoteProcess'),'serviceName']


# 2 fault path
#fault = pd.read_csv(r'E:\AiOps\AIOps_datasets_2020\故障整理（预赛）.csv')
#fault['start_time'] = pd.to_datetime(fault['start_time']) #标准数据
#fault['log_time'] = pd.to_datetime(fault['log_time']) #标准数据

# 3 调用链时序分析
# 参数anomaly_id表示duration：取异常点前后2min30s 的数据分析, traceId 1000+,  取异常点前后1min 的数据分析, traceId 500+
#anomaly_id = 2  # para1:fault的index
start_time = '2020-04-21 00:47:00'
#start_time = pd.to_datetime(start_time) 
start_time = datetime_timestamp(start_time)
print("anomaly start at ",start_time)
#duration = pd.Timedelta(days=0, minutes=2, seconds=0)
duration = 2 * 60*1000
all_trace_duration = all_trace.loc[ (all_trace['startTime'] >= start_time - duration) &
                                   (all_trace['startTime'] < start_time + duration), : ]

# CSF特殊处理
#def fill_newServiceName(id_para, col):
#    try:
#        return all_trace_duration.loc[(all_trace_duration.pid == id_para), col].values[0]
#    except:
#        return np.NAN
# 对于CSF类的需要结合子调用，使用子调用的cmbdid作为本次调用的serviceName， all_trace_duration小点，跑快点
#all_trace_CSF = all_trace_duration.loc[(all_trace_duration.callType == 'CSF'),:]
#all_trace_CSF['newServiceName'] = all_trace_CSF.apply(lambda row: fill_newServiceName(row['id'], 'cmdb_id'), axis = 1)

traceIds = all_trace_duration['traceId'].value_counts().to_frame()

# 参数 all_trace_duration的前 n_chain 个调用链, 分析主要调用链省时间
n_chain = 100  # para2 调用链数目

for traceId in traceIds.head(n_chain).index:
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
all_trace_duration_top_traceIds.shape

def func_pid(pid, col):
    try:
        return all_trace_duration_top_traceIds.loc[all_trace_duration_top_traceIds['id'] == pid, col].values[0]
    except:
        return np.NAN

all_trace_duration_top_traceIds['pid_newServiceName'] = \
all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'],'newServiceName'), axis=1)

all_trace_duration_top_traceIds['pid_cmdb_id'] = \
all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'],'cmdb_id'), axis=1)

all_trace_duration_top_traceIds['pid_serviceName'] = \
all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'],'serviceName'), axis=1)

all_trace_duration_top_traceIds['pid_dsName'] = \
all_trace_duration_top_traceIds.apply(lambda row: func_pid(row['pid'],'dsName'), axis=1)


# 计算异常分数矩阵
# 双层遍历1： cmdb_id	serviceName	dsName 的时序图看看情况
pid_col_name =  'pid_cmdb_id' #'pid_newServiceName'
id_col_name =  'cmdb_id' #'newServiceName'

#pid_col_name =  'pid_newServiceName'
#id_col_name =  'newServiceName'

interval_point = 5

# 每行是pid， 列对应这 id，被调用的
pid_cmdb_names = all_trace_duration_top_traceIds[pid_col_name].unique()
id_cmdb_names = all_trace_duration_top_traceIds[id_col_name].unique()

#df_scores = pd.DataFrame(index = pd.Index(pid_cmdb_names.tolist()),
#                        columns = pd.Index(id_cmdb_names.tolist()))

df_series = pd.DataFrame(index = pd.Index(pid_cmdb_names.tolist()),
                        columns = pd.Index(id_cmdb_names.tolist()))

array_series = np.array(df_series)

k=0
#for pid_cmdb in pid_cmdb_names:
#    for id_cmdb in id_cmdb_names:
for i in range(len(pid_cmdb_names)):
    for j in range(len(id_cmdb_names)):
        k+=1
        # 取出 col_name 里面取值为 cmdb 的调用
        all_trace_duration_top_traceIds_cmdb = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds[pid_col_name] == pid_cmdb_names[i]) & \
                                            (all_trace_duration_top_traceIds[id_col_name] == id_cmdb_names[j])]
        if(all_trace_duration_top_traceIds_cmdb.shape[0] < 2 * interval_point):
            continue;
        # 按时间排序
        all_trace_duration_top_traceIds_cmdb.sort_values(by=['startTime'],ascending=True,inplace=True)
        #current_df_series = all_trace_duration_top_traceIds_cmdb.loc[:,['startTime','elapsedTime']]
        print(k)
        array_series[i,j] =  all_trace_duration_top_traceIds_cmdb
        #df_series.loc[pid_cmdb, id_cmdb] = current_df_series

#print(df_series)

'''
for pid_cmdb in pid_cmdb_names:
    for id_cmdb in id_cmdb_names:

        # 取出 col_name 里面取值为 cmdb 的调用
        all_trace_duration_top_traceIds_cmdb = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds[pid_col_name] == pid_cmdb) & \
                                            (all_trace_duration_top_traceIds[id_col_name] == id_cmdb)]
        if(all_trace_duration_top_traceIds_cmdb.shape[0] < 2 * interval_point):
            continue;
        # 按时间排序
        all_trace_duration_top_traceIds_cmdb.sort_values(by=['startTime'],ascending=True,inplace=True)
        # anomaly score calculate
        cmdb_avg_ElaTime = all_trace_duration_top_traceIds_cmdb.groupby('startTime')['elapsedTime'].mean().to_frame()
        series_beforeFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index < start_time]['elapsedTime'].tail(interval_point).values
        series_afterFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index >= start_time]['elapsedTime'].head(interval_point).values
        # 指标这个可以在看看用什么，这里用的故障后n分钟的平均耗时 - 故障前的n分钟平均耗时
        anomaly_score = (series_afterFalut.mean() - series_beforeFalut.mean()) / series_beforeFalut.mean()
        df_scores.loc[pid_cmdb, id_cmdb] = anomaly_score

print(df_scores)
'''

ll = array_series[0,0]

def unify_ts(df, fault_time, duration, window_size ):
    """
    

    Parameters
    ----------
    df : TYPE           DataFrame
        DESCRIPTION.    里面包含两列['startTime','elapsedTime']
    fault_time : TYPE   Timestamp
        DESCRIPTION.    发生错误的时间
    duration : TYPE     int
        DESCRIPTION.    总时间的长度
        Example         2*60*1000      （前后各取） 两分钟
    window_size : TYPE  int
        DESCRIPTION.    时间窗的长度
        Example         1000      （前后各取一秒）
        
    Returns
    -------
    None.

    """
    start_time = fault_time - duration
    end_time = fault_time + duration
    existing_time_list = list(set(df['startTime'].to_list()))
    
    interval = 1000
    
    current_time = start_time
    time_list = []
    elapsedTime_list = []
    
    df_startTime_list = df['startTime'].to_list()
    df_elapsedTime_list = df['elapsedTime'].to_list()
    start_elapsed_dict = dict(zip(df_startTime_list,df_elapsedTime_list))
    
    while current_time < end_time:
        time_list.append(current_time)
#        window = (current_time - window_size) & (current_time + window_size) 
        df_window = df.loc[ (df['startTime'] >= current_time - window_size) &
                                   (df['startTime'] < current_time + window_size), : ]
        if df_window.shape[0]!=0:
            MA_elapsedTime = np.mean(df_window['elapsedTime'])
            elapsedTime_list.append(MA_elapsedTime)
        else:
            array = np.array(existing_time_list)
            idx = (np.abs(array-current_time)).argmin()
            nearest_time = array[idx]
            nearest_elapsedTime = start_elapsed_dict[nearest_time]
            elapsedTime_list.append(nearest_elapsedTime)
        current_time += interval
        
    df_new = pd.DataFrame({'startTime':time_list,'elapsedTime':elapsedTime_list})
    
    return df_new

new_ll = unify_ts(ll, start_time, duration, window_size = 1000 )
      

############################## 调试
 
fault_time = start_time
window_size = 1000
df = ll 

start_time = fault_time - duration
end_time = fault_time + duration
existing_time_list = list(set(df['startTime'].to_list()))

interval = 1000

current_time = start_time
time_list = []
elapsedTime_list = []  

df_startTime_list = df['startTime'].to_list()
df_elapsedTime_list = df['elapsedTime'].to_list()
start_elapsed_dict = dict(zip(df_startTime_list,df_elapsedTime_list))

while current_time < end_time:
    time_list.append(current_time)
#        window = (current_time - window_size) & (current_time + window_size) 
    df_window = df.loc[ (df['startTime'] >= current_time - window_size) &
                               (df['startTime'] < current_time + window_size), : ]
    if df_window.shape[0]!=0:
        MA_elapsedTime = np.mean(df_window['elapsedTime'])
        elapsedTime_list.append(MA_elapsedTime)
    else:
        array = np.array(existing_time_list)
        idx = (np.abs(array-current_time)).argmin()
        nearest_time = array[idx]
        nearest_elapsedTime = start_elapsed_dict[nearest_time]
        elapsedTime_list.append(nearest_elapsedTime)
    current_time += interval
        