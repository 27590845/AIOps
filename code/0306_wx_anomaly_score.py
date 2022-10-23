import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
#%matplotlib inline
from graphviz import Digraph
from IPython.display import display, Image
import os
import math
from wand.image import Image
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import random
import scipy

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import requests

import warnings
warnings.filterwarnings('ignore')

#dates = ['2020_04_11', '2020_04_20', ,'2020_04_22', '2020_04_23', '2020_04_24', '2020_04_25', '2020_04_26',
#         '2020_05_22', '2020_05_23', '2020_05_24', '2020_05_25', '2020_05_26', '2020_05_27', '2020_05_28', '2020_05_29', '2020_05_30', '2020_05_31']

#dates = ['2020_04_11', 2020_04_21', '2020_04_22', '2020_04_23', '2020_04_24', '2020_04_25', '2020_04_26']
dates = ['2020_04_20']
for path_date in dates:
    #path_date = '2020_04_26'
    fault = pd.read_excel('/Users/wangxue/gitpro/20200307AIOps/官方AIOps挑战赛数据/故障整理all.xlsx')
    fault['start_time'] = pd.to_datetime(fault['start_time']) #标准数据
    #fault.head(1)
    path = '/Users/wangxue/gitpro/20200307AIOps/官方AIOps挑战赛数据/AIOps挑战赛数据/' + path_date + '/调用链指标/'
    trace_csf = pd.read_csv(path + 'trace_csf.csv')
    trace_fly_remote = pd.read_csv(path +'trace_fly_remote.csv')
    trace_jdbc = pd.read_csv(path + 'trace_jdbc.csv')
    trace_local = pd.read_csv(path + 'trace_local.csv')
    trace_osb = pd.read_csv(path +'trace_osb.csv')
    trace_remote_process = pd.read_csv(path + 'trace_remote_process.csv')
    # concat all
    all_trace = pd.concat([trace_csf, trace_fly_remote, trace_jdbc, trace_local, trace_osb, trace_remote_process], axis=0, sort=False)
    all_trace.reset_index(drop=True, inplace=True)
    # time format
    all_trace["startTime"] = all_trace["startTime"].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
    all_trace["startTime"] = pd.to_datetime(all_trace["startTime"])

    esb = pd.read_csv('/Users/wangxue/gitpro/20200307AIOps/官方AIOps挑战赛数据/AIOps挑战赛数据/' +path_date+ '/业务指标/esb.csv')

    esb['startTime'] = esb["startTime"].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000)\
                                       .strftime('%Y-%m-%d %H:%M:%S'))
    esb["startTime"] = pd.to_datetime(esb["startTime"])

    # esb.head(5)
    fault_date = fault.loc[(fault['start_time'] < pd.datetime(2020,int(path_date.split('_')[1]),
                                                              int(path_date.split('_')[2]),
                                                              23,59,59)) &
                            (fault['start_time'] > pd.datetime(2020,int(path_date.split('_')[1]),
                                                               int(path_date.split('_')[2]),0,0,0)),:]
    print("fault date shape = ", fault_date.shape)




    # 一次异常分数计算
    cmdb_names = ['docker_001','docker_002','docker_003','docker_004',
                  'docker_005', 'docker_006', 'docker_007','docker_008',
                 'db_003','db_007','db_009',
                 'os_001','os_002','os_003','os_004','os_005','os_006','os_007','os_008','os_009','os_010','os_011',
                 'os_012','os_013','os_014','os_015','os_016','os_017','os_018','os_019','os_020','os_021','os_022']
    # 拼接所有的fault的 trace 时序 特征
    for cmdb_pid in cmdb_names:
        for cmdb_id in cmdb_names:
            fault_date.insert(fault_date.shape[1], cmdb_pid+'_'+cmdb_id, 0.0)
    # 方便存Anomaly score2
    fault_date1 = fault_date

    for anomaly_id in fault_date.index.to_list():
        #anomaly_id = fault_date.index.to_list()[2]  # fault csv的index
        start_time = fault_date.at[anomaly_id,'start_time']
        print(anomaly_id, start_time)
        # 可调参数minutes
        duration = pd.Timedelta(days=0, minutes=5, seconds=0)

        all_trace_duration = all_trace.loc[ (all_trace['startTime'] > start_time - duration) &
                                           (all_trace['startTime'] <= start_time + duration), : ]

        all_traceIds = list(all_trace_duration['traceId'].unique())
        # 可调参数，具体的调用链的条数 top_traceIds
        #random.seed(2021)
        #top_traceIds = random.sample(all_traceIds, int(0.9 * len(all_traceIds)) )
        top_traceIds = list(all_traceIds)
        all_trace_duration_top_traceIds = all_trace_duration.loc[all_trace_duration['traceId'].isin(top_traceIds)]

        for traceId in top_traceIds:
            # 同一个traceid的一次整体全部调用, 减去子调用的耗时才是真实调用的耗时
            oneCall = all_trace_duration_top_traceIds.loc[all_trace_duration_top_traceIds['traceId'] == traceId]
            for unq_id in oneCall['id'].unique():
                unq_id_child_elapsedTime = 0
                try:
                    unq_id_child_elapsedTime = oneCall[oneCall['pid'] == unq_id]['elapsedTime'].sum()
                except:
                    print("unq_id_child_elapsedTime 0")
                unq_id_index = oneCall.loc[oneCall['id'] == unq_id].index.tolist()[0]
                all_trace_duration_top_traceIds.at[unq_id_index, 'elapsedTime'] -= unq_id_child_elapsedTime


        # 处理ServiceName为真正调用方
        all_trace_duration_top_traceIds.insert(all_trace_duration.shape[1], 'newServiceName', np.NAN)
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'JDBC'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'JDBC'),'dsName']

        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'LOCAL'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'LOCAL'),'dsName']

        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'OSB'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'OSB'),'cmdb_id']

        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'RemoteProcess'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'RemoteProcess'),'cmdb_id']

        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'FlyRemote'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'FlyRemote'),'cmdb_id']

        def fill_newServiceName(id_para, col):
            try:
                return all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.pid == id_para), col].values[0]
            except:
                return np.NAN
        # 对于CSF类的需要结合子调用，使用子调用的cmbdid作为本次调用的serviceName， all_trace_duration_top_traceIds少快点
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'CSF'),'newServiceName'] = \
        all_trace_duration_top_traceIds.loc[(all_trace_duration_top_traceIds.callType == 'CSF'),:].apply(lambda row: fill_newServiceName(row['id'], 'cmdb_id'), axis = 1)


        # 双层遍历1： cmdb_id	serviceName	dsName 的时序图看看情况
        # pid_col_name =  'pid_cmdb_id' #'pid_newServiceName'
        # id_col_name =  'cmdb_id' #'newServiceName'
        pid_col_name =  'cmdb_id'
        id_col_name =  'newServiceName'
        interval_point = 20

        # 每行是pid， 列对应这 id，被调用的
        pid_cmdb_names = all_trace_duration_top_traceIds[pid_col_name].unique()
        id_cmdb_names = all_trace_duration_top_traceIds[id_col_name].unique()

        df_scores = pd.DataFrame(index = pd.Index(pid_cmdb_names.tolist()),
                                 columns = pd.Index(id_cmdb_names.tolist()))

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
                #cmdb_avg_ElaTime = all_trace_duration_top_traceIds_cmdb
                cmdb_avg_ElaTime = all_trace_duration_top_traceIds_cmdb.groupby('startTime')['elapsedTime'].mean().to_frame()
                #series_beforeFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index < start_time]['elapsedTime'].tail(interval_point).values
                #series_afterFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index >= start_time]['elapsedTime'].head(interval_point).values
                series_beforeFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index < start_time]['elapsedTime'].values
                series_afterFalut = cmdb_avg_ElaTime.loc[cmdb_avg_ElaTime.index >= start_time]['elapsedTime'].values
                before_len = len(series_beforeFalut)
                after_len = len(series_afterFalut)
                if (before_len > after_len): #对齐数据
                    series_beforeFalut = series_beforeFalut[ before_len - after_len : before_len ]
                else:
                    series_afterFalut = series_afterFalut[0 : before_len]

                #anomaly_score = (series_afterFalut.mean() - series_beforeFalut.mean()) / series_beforeFalut.mean()
                # 避免0值问题
                #series_beforeFalut = [0.01 if x == 0.0 else x for x in series_beforeFalut]
                #series_afterFalut = [0.01 if x == 0.0 else x for x in series_afterFalut]
                anomaly_score = scipy.stats.entropy(series_beforeFalut, series_afterFalut)
                df_scores.loc[pid_cmdb, id_cmdb] = anomaly_score
                # 行是index，即anomaly_id； 列是cmdb_pid+'_'+cmdb_id
                fault_date.at[anomaly_id, pid_cmdb+'_'+id_cmdb] = anomaly_score

                M = (series_beforeFalut + series_afterFalut) / 2
                anomaly_score1 = 0.5*scipy.stats.entropy(series_beforeFalut, M)+0.5*scipy.stats.entropy(series_afterFalut, M)
                fault_date1.at[anomaly_id, pid_cmdb+'_'+id_cmdb] = anomaly_score1

        print(fault_date.at[anomaly_id,'name'])
        #print(df_scores)
        
    # %%
    fault_date.to_csv('../faults/fault_' +path_date+ '.csv', index=0)
    fault_date1.to_csv('../faults1/fault_' +path_date+ '.csv', index=0)
