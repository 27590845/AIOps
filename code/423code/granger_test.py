import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from minepy import MINE
from statsmodels.tsa.stattools import grangercausalitytests


# 取一段时间窗的指标：type:'itemid'/'cmdb_name'，iid:itemid的int/[cmdb_id,name]
def find_timeseries_interval(df, iid, timestart=None, timeend=None, coltype="itemid"):
    """
    功能：根据给定的dataframe, iid, timestart, timeend, 从dataframe中找出<=timeend 且>=timestart的所有timestamp和对应的value

    参数：
    df: dataframe,至少包括列:itemid, timestamp, value
    iid:当type=='itemid'时，为需要寻找的itemid值，int类型(对于str类型代码中做了int转换)；当flag=='cmdb_name'时，为需要寻找的cmdb_id+name值，
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


def findall_cmdb_item_name(df_new):
    """
    作用：给定已选定有关类型指标名称的dataframe，将其中所有的（容器，指标id，指标名）三元组存入list
    参数：
    df_new: 已选定指标名称的dataframe
    返回：
    uniindex_list: (cmdb_id,itemid,name)三元组list

    例：关于cpu指标的dataframe，输出
    [('docker_001', 999999996381357, 'container_cpu_used'),
     ('docker_002', 999999996381380, 'container_cpu_used'),
     ('docker_003', 999999996381403, 'container_cpu_used'),
     ('docker_004', 999999996381449, 'container_cpu_used'),
     ('docker_005', 999999996381265, 'container_cpu_used'),
     ('docker_006', 999999996381288, 'container_cpu_used'),
     ('docker_007', 999999996381311, 'container_cpu_used'),
     ('docker_008', 999999996381334, 'container_cpu_used')]
    """
    # 找到所有的itemid，以及对应的cmdb_id和name
    uniindex_list = list(df_new.groupby(["cmdb_id", "itemid", "name"]).nunique().index)
    return uniindex_list


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
        return 1
    else:
        return 0


def same_gap_adj(met1, met2):
    # # 1. 截时间：从后开始的开始，从先结束的结束
    # 开始晚的
    met1_s = met1.iloc[0, 0]
    met2_s = met2.iloc[0, 0]
    met_s = max(met1_s, met2_s)

    # 结束早的
    met1_e = met1.iloc[-1, 0]
    met2_e = met2.iloc[-1, 0]
    met_e = min(met1_e, met2_e)

    # 截断时间
    df_met1 = met1[met1['timestamp'] >= met_s]
    df_met1 = df_met1[df_met1['timestamp'] <= met_e].reset_index(drop=True)
    df_met2 = met2[met2['timestamp'] >= met_s]
    df_met2 = df_met2[df_met2['timestamp'] <= met_e].reset_index(drop=True)

    # # 2. 判断初始时间距离>1/2gap ，取另一条开始时间
    gap = df_met1.iloc[1, 0] - df_met1.iloc[0, 0]
    s1 = df_met1.iloc[0, 0]
    s2 = df_met2.iloc[0, 0]
    s = [s1, s2]
    if np.abs(s1-s2) > 1/2*gap:
        if s1 == met_s:
            df_met1 = df_met1.drop([0]).reset_index(drop=True)
        else:
            df_met2 = df_met2.drop([0]).reset_index(drop=True)

    # # 3. 判断长度不相等，长的扔掉最后一个
    n1 = len(df_met1)
    n2 = len(df_met2)
    if n1 > n2:
        df_met1 = df_met1.drop([n1-1])
    elif n1 < n2:
        df_met2 = df_met2.drop([n2-1])

    return df_met1, df_met2


def ts_adjust(met1, met2):
    met1 = met1.reset_index(drop=True)

    met2 = met2.reset_index(drop=True)

    met = [met1, met2]
    # 1.间隔大的补齐：补中间
    gap1 = met[0].iloc[1, 0] - met[0].iloc[0, 0]
    gap2 = met[1].iloc[1, 0] - met[1].iloc[0, 0]
    gap = [gap1, gap2]
    gap_min = min(gap[0], gap[1])
    if gap[0] / gap_min > 2 or gap[1] / gap_min > 2:
        for i in range(2):
            if gap[i] != gap_min:
                # 重新generate时间戳
                start = met[i].iloc[0, 0]
                stop = met[i].iloc[-1, 0]
                number = int(round((stop - start) / gap_min)) + 1
                t = np.linspace(start, stop, num=number, endpoint=True)

                # 补齐，和前面一个值相同
                fill_n = int(round(gap[i] / gap_min))
                ts = np.repeat(np.array(met2['value']), fill_n)
                ts = ts[:-(fill_n - 1)]

                # 重开一个df
                met_new = pd.DataFrame(columns=['timestamp', 'value'])
                met_new['timestamp'] = t
                met_new['value'] = ts

                met_same = met[1 - i]
                met[i] = met_new
                met[1 - i] = met_same

    df_met1, df_met2 = same_gap_adj(met[0], met[1])

    return df_met1, df_met2

def is_const(met):
    set_value = set(list(met))
    if len(set_value) == 1:
        return True
    else:
        return False
    
def is_wukejiuyao(met):
    a = list(met)
    gap = []
    for i in range(len(a)-1):
        gap.append(a[i+1] - a[i])
    set_value = set(gap)
    if len(set_value) > 9:
        return True
    else:
        return False
    
filedir = r"E:\AiOps\AIOps_datasets_2020"
os_filedir_23 = filedir + "\\2020_04_23\平台指标\os_linux.csv"
df_os_23 = pd.read_csv(os_filedir_23)

df = df_os_23[df_os_23['cmdb_id'] == 'os_021']

# 循环提取两条时间序列
triple = findall_cmdb_item_name(df)
n = len(triple)
granger_mat = np.eye(n, k=0, dtype=int)

for i in range(n):
    ts1 = find_timeseries_interval(df, triple[i][1], timestart=None, timeend=None, coltype="itemid")
    if is_const(ts1['value']):
        for j in range(i+1, n):
            granger_mat[i][j] = 3
            granger_mat[j][i] = 3

    elif is_wukejiuyao(ts1['timestamp']):
        for j in range(i+1, n):
            granger_mat[i][j] = 4
            granger_mat[j][i] = 4
    else:
        for j in range(i+1, n):
            print(i, str(triple[i]))
            print(j, str(triple[j]))
            ts2 = find_timeseries_interval(df, triple[j][1], timestart=None, timeend=None, coltype='itemid')
            if is_const(ts2['value']):
                granger_mat[i][j] = 3
                granger_mat[j][i] = 3
            elif is_wukejiuyao(ts2['timestamp']):
                granger_mat[i][j] = 4
                granger_mat[j][i] = 4
            else:
                try:
                    met1, met2 = ts_adjust(ts1, ts2)
                    granger_mat[i][j] = granger(met1[['value']], met2[['value']])
                    granger_mat[j][i] = granger(met2[['value']], met1[['value']])
                except:
                    granger_mat[i][j] = 2
                    granger_mat[j][i] = 2



pd.DataFrame(granger_mat).to_csv('error.csv')




