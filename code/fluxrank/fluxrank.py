
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
#%matplotlib inline
import os
import math
import random
import scipy
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN


def digest_distillation(cmdb_features):
    """
    作用：
    对不同的cmdb进行分类，这里docker，os，db分开；用dbscan分类，距离指标为pearson

    参数：
    cmdb_features: 每个cmdb的特征(这里注意docker,os,db分开), np.array
    例：有2个cmdb：os_1, os_2，3个指标：kpi_1,kpi_2,kpi_3
    cmdb_features = [[o_11,u_11,o_12,u_12,o_13,u_13], [o_21,u_21,o_22,u_22,o_23,u_23]]
    其中o_ij：cmdb i中指标j的overflow概率；u_ij：cmdb i中指标j的underflow概率

    返回：
    digests: np.array, 每个cmdb所属的类别
    例：有9个cmdb，digests = [0,1,2,1,2,3,3,2,0]
    共4类，第i个cmdb的类别对应于digests[i]
    """
    clustering = DBSCAN(eps=1, min_samples=2, metric='precomputed')
    X = pd.DataFrame(cmdb_features).T
    distance_matrix = X.corr(method='pearson')
    digests = clustering.fit(abs(distance_matrix)).labels_

    k = max(digests)
    for i in range(len(digests)):
        if digests[i] == -1:
            digests[i] = k+1
            k+=1
    return digests

# 测试函数
cmdb_features =  np.random.random((6,6))
digests = digest_distillation(cmdb_features)


一开始先建立两个字典
dict_cmdb：把cmdb和一个数字对应，后面无论是df还是array中的cmdb的顺序全部按照字典标号决定。
dict_kpi：把kpi和一个数字对应，后面无论是df还是array中的kpi的顺序全部按照字典标号决定。
1 Change Qualification
change_start_timestamp(times, values, etimestamp)
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
def change_start_timestamp(times, values, etimestamp):
    """
    作用：
    针对给定的一段时间序列和错误时间点，一阶差分后用3-sigma法（这里取2.5-sigma）找到异常开始时间，
    对于用3-sigma无法确定异常发生时间的时间序列（异常不明显或根本没有），
    找其偏离均值最大的点作为异常开始时间，顺便返回KDE之后的overflow和underflow值
    参数：
    times: 时间序列的timestamp数列，np.array
    values: 时间序列的value数列，np.array
    etimestamp: 时间序列的error_timestamp，int
    返回：
    一个list：3个元素
    [change_start: 异常开始时间戳-etimestamp，int
    po: overflow 概率, float
    pu: underflow 概率, float]
    """
    if len(values)>1:
        # 一阶差分
        times = times[1:]
        div_values = values[1:] - values[:values.shape[0] - 1]
    
        # 3-sigma法则
        std_values = np.std(div_values)
        mean_values = np.mean(div_values)
        up_thres = mean_values + 2.5 * std_values
        adjusted_div_values = abs(div_values - mean_values)
        bool_array = adjusted_div_values > up_thres
        # plt.plot(times,div_values)
    
        # 如果没有可能异常点，返回0
        if sum(bool_array) == 0:
            # 返回第一个最大值所在的位置下标
            pos = np.argmax(adjusted_div_values)
            change_start_time = times[pos]
        else:
            # 从可疑点中找距离error_timestamp最近的时间点(此处可能会有问题，如果最近的是change end time怎么办？？？)
            minindex = np.argmin(abs(times[bool_array] - etimestamp))
            change_start_time = times[bool_array][minindex]
    
        if len(times[times < change_start_time]) >= 2 and len(times[times > change_start_time]) >= 2:
            po, pu = kde(times, div_values, change_start_time)
            return [change_start_time-etimestamp, po, pu]
    return [0, 0, 0]
# 测试change_start_time 函数
times = np.array([1,2,3,4,5,6,7])
X = np.array([1,2,1,20,1,2,1])
etimestamp = 4
change_start_timestamp(times, X, etimestamp)
中间函数：kde(times, X, change_start_time)
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
def kde(times, X, change_start_time):
    """
    作用：用KDE方法计算某条时间序列change_start_time后overflow和underflow的概率
    参数：
    times: timestamp序列，array
    X: times对应的值序列，array
    change_start_time: 变化开始时间，timestamp，注意这里的change_start_time是绝对时间
    返回：
    po: overflow概率
    pu: underflow概率
    """
    # 找到times里面对应change_start_time的下标
    pos = np.argwhere(times == change_start_time)[0][0]
    # bandwidth
    width = 0.1
    # X标准化，[0,1]之间
    X_min = np.min(X)
    X_max = np.max(X)
    X_std = (X - X_min) / (X_max - X_min)
    X_train = X_std[:pos]
    X_test = X_std[pos:]
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
    return po, pu
# 测试kde函数
times = np.array([1,2,3,4,5,6,7])
X = np.array([1,2,1,20,1,2,1])
change_start_time = 4
kde(times, X,change_start_time)
2 Digest Distillation 
运行下面的函数之前前面要对n个cmdb，m个指标循环计算change qualification里面的指标。
digest_distillation(cmdb_features)
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
def digest_distillation(cmdb_features):
    """
    作用：
    对不同的cmdb进行分类，这里docker，os，db分开；用dbscan分类，距离指标为pearson
    参数：
    cmdb_features: 每个cmdb的特征(这里注意docker,os,db分开), np.array
    例：有2个cmdb：os_1, os_2，3个指标：kpi_1,kpi_2,kpi_3
    cmdb_features = [[o_11,u_11,o_12,u_12,o_13,u_13], [o_21,u_21,o_22,u_22,o_23,u_23]]
    其中o_ij：cmdb i中指标j的overflow概率；u_ij：cmdb i中指标j的underflow概率
    返回：
    digests: np.array, 每个cmdb所属的类别
    例：有9个cmdb，digests = [0,1,2,1,2,3,3,2,0]
    共4类，第i个cmdb的类别对应于digests[i]
    """
    clustering = DBSCAN(eps=1, min_samples=2, metric='precomputed')
    X = pd.DataFrame(cmdb_features).T
    distance_matrix = X.corr(method='pearson')
    digests = clustering.fit(abs(distance_matrix)).labels_
    k = max(digests)
    for i in range(len(digests)):
        if digests[i] == -1:
            digests[i] = k+1
            k+=1
    return digests
# 测试函数
cmdb_features =  np.random.random((6,6))
digests = digest_distillation(cmdb_features)
3 Digest Ranking
label_digest(df_cmdb,digests, dict_cmdb)
import pandas as pd
import numpy as np
def label_digest(df_cmdb,digests, dict_cmdb):
    """
    作用：
    查找真实故障编号对应故障根因所属的类别，作为logistic模型训练的标签
    参数：
    df_cmdb: dataframe, 里面要有['index', 'cmdb']两列，其中'index'是故障编号，'cmdb'是故障对应的根因网元。
    digests: 字典，key：故障编号index, value: np.array,故障编号对应的所有网元的分类，digest_distillation函数中的返回格式，表示每个cmdb所属的类
    dict_cmdb: 字典，给每个网元编号，主要用来查找网元名称对应的是digests中的哪个位置
    返回：
    df_label_digest: 包含故障编号和所有digest的dataframe，并判断该故障编号下根因是否在这个digest中，列名['index', 'digest','is_root']
    """
    df = df_cmdb.copy(deep=True)
    df['label_digest'] = df.apply(lambda x: digests.get(x['index'])[dict_cmdb.get(x['cmdb'])], axis=1)
    # 拆分
    fault_index = list(df['index'])
    new_list = []
    for index in fault_index:
        temp_df = df[df['index']==index]
        
        digests_cnt = len(list(set(digests.get(index))))
        for i in range(digests_cnt):
            if temp_df['label_digest'].values[0]== i:
                is_root = 1
    #            print(is_root)
            else:
                is_root=0
    #            print(is_root)
            new_list.append([index, i, is_root])
    
    df_label_digest = pd.DataFrame(new_list, columns =['index', 'digest', 'is_root'])
    return df_label_digest
    
    
#测试函数
filedir = r'E:\My source\02   Project\2   Anormaly Detection and Root Cause Location\2   AIOps Challenge\AIOps挑战赛数据'
filename = '\故障整理all.xlsx'
df_fault = pd.read_excel(filedir+filename).iloc[:2, :] # temp
df_cmdb = df_fault[['index', 'name']]
df_cmdb.columns = ['index', 'cmdb']
dict_cmdb = {'docker_003':0, 'docker_002':1, 'docker_001':2}
digests = {1:np.array([0,1,1]), 2:np.array([1,2,0])}
label_digest(df_cmdb, digests, dict_cmdb)




def digest_features(change_start_arr, o_arr, u_arr, digests):
    """
    作用：
    计算每个digest中所有change_start_arr绝对值小于2分钟（暂定）的KPI的
    change_start的 max, min, sum, mean, std; 
    overflow , underflow的max, min, sum,mean 构成每个digest的特征
    
    参数：
    change_start_arr: np.array, shape: cmdb个数*每个cmdb的指标个数
    例：2个cmdb，3个指标
    change_start_arr = [[s11,s12,s13], [s21,s22,s23]], 
                        sij: 第i个cmdb的第j个指标的change_start_time （这里的change_start_time是个相对时间）
    o_arr: np.array, shape: cmdb个数*每个cmdb的指标个数
    u_arr: np.array, shape: cmdb个数*每个cmdb的指标个数
    digests: digest_distillation函数中的返回格式，表示每个cmdb所属的类
    
    返回：
    digest_features_df: DataFrame，shape = digest个数*14(digest名称+feature个数)
    """
    
    
    cmdb_num = change_start_arr.shape[0]               #有多少个cmdb
    #cmdb_id_num = change_start_arr.shape[1]
    
    # 先求出每个cmdb的特征
    cmdb_features_df = pd.DataFrame()
    for i in range(cmdb_num):
        change_start_arr_i = change_start_arr[i]
        o_arr_i = o_arr[i]
        u_arr_i = u_arr[i]
        
        # 如果该cmdb中所有KPI的change start time均为 0, 就放弃这个cmdb
        if len(set(change_start_arr_i)) == 1 and change_start_arr_i[0] == 0:
            continue
        
        # 将该cmdb中所有小于两分钟的change start time取出来
        change_start_arr_i_dict = dict(zip(range(len(change_start_arr_i)),change_start_arr_i))
        change_start_arr_i_dict_lessthan2 = \
        {key: value for key, value in change_start_arr_i_dict.items() if abs(value) < 2*60*1000 and abs(value)>0}
        
        # 如果这个cmdb满足要求的KPI为空，则放弃这个cmdb
        if len(change_start_arr_i_dict_lessthan2) == 0:
            continue
        
        # 提取该cmdb满足要求的KPI
        change_start_arr_i_new = list(change_start_arr_i_dict_lessthan2.values())
        o_arr_i_new = []
        u_arr_i_new = []
        
        for k in list(change_start_arr_i_dict_lessthan2.keys()):
            o_arr_i_new.append(o_arr_i[k])
            u_arr_i_new.append(u_arr_i[k])
        
        cmdb_features_df.loc[i,'digest'] = digests[i]
        
        cmdb_features_df.loc[i, 'change_start_max'] = max(change_start_arr_i_new)
        
        cmdb_features_df.loc[i, 'change_start_min'] = min(change_start_arr_i_new)
        cmdb_features_df.loc[i, 'change_start_sum'] = sum(change_start_arr_i_new)
        cmdb_features_df.loc[i, 'change_start_mean'] = np.mean(change_start_arr_i_new)
        cmdb_features_df.loc[i, 'change_start_std'] = np.std(change_start_arr_i_new)
        
        cmdb_features_df.loc[i, 'o_max'] = max(o_arr_i_new)
        cmdb_features_df.loc[i, 'o_min'] = min(o_arr_i_new)
        cmdb_features_df.loc[i, 'o_sum'] = sum(o_arr_i_new)
        cmdb_features_df.loc[i, 'o_mean'] = np.mean(o_arr_i_new)
        
        cmdb_features_df.loc[i, 'u_max'] = max(u_arr_i_new)
        cmdb_features_df.loc[i, 'u_min'] = min(u_arr_i_new)
        cmdb_features_df.loc[i, 'u_sum'] = sum(u_arr_i_new)
        cmdb_features_df.loc[i, 'u_mean'] = np.mean(u_arr_i_new)
        
    # 求出每个 digest 的特征
    digest_features_df = pd.DataFrame()
    grouped_digest = cmdb_features_df.groupby('digest')
    i = 0
    for digest, group in grouped_digest:
        
        digest_features_df.loc[i,'digest'] = digest
        digest_features_df.loc[i, 'change_start_max'] = max(group['change_start_max'])
        digest_features_df.loc[i, 'change_start_min'] = min(group['change_start_min'])
        digest_features_df.loc[i, 'change_start_sum'] = sum(group['change_start_sum'])
        digest_features_df.loc[i, 'change_start_mean'] = np.mean(group['change_start_mean'])
        digest_features_df.loc[i, 'change_start_std'] = np.std(group['change_start_std'])
        
        digest_features_df.loc[i, 'o_max'] = max(group['o_max'])
        digest_features_df.loc[i, 'o_min'] = min(group['o_min'])
        digest_features_df.loc[i, 'o_sum'] = sum(group['o_sum'])
        digest_features_df.loc[i, 'o_mean'] = np.mean(group['o_mean'])
        
        digest_features_df.loc[i, 'u_max'] = max(group['o_max'])
        digest_features_df.loc[i, 'u_min'] = min(group['o_min'])
        digest_features_df.loc[i, 'u_sum'] = sum(group['o_sum'])
        digest_features_df.loc[i, 'u_mean'] = np.mean(group['u_mean'])
        i +=1
        
    
    return digest_features_df
    
# 测试函数
# 6个cmdb,6个 KPI
change_start_arr = np.random.randint(1,5,size=[6,6])
o_arr =  np.random.random(((6,6)))
u_arr =  np.random.random(((6,6)))
digests = np.array([0,1,2,3,1,2])
df = digest_features(change_start_arr, o_arr, u_arr, digests)
digest_rank(o_dict, u_dict)

def digest_rank(o_dict, u_dict):
    '''
    dict_kpi：把kpi和一个数字对应，后面无论是df还是array中的kpi的顺序全部按照字典标号决定。
    参数:
    o_dict: dict,某一个digest中的所有kpi的overflow概率，shape = 所有kpi的个数
            {itemid1:o1, itemid2:o2}
    u_dict: dict,某一个digest中的kpi的underflow概率
    返回：
    prob_rank_dict: dict, v = max{o,u}的值排序 shape = 所有kpi的个数
            {itemid2:v2, itemid1:v1}
    '''
    prob_rank_dict = {}
    for key, value in o_dict.items():
        prob_rank_dict[key] = max(value, u_dict[key])
    prob_rank_dict = dict(sorted(prob_rank_dict.items(), key=lambda d: d[1]))
    return prob_rank_dict
        
    
# 测试函数
o_dict = {'a': 5, 'b':2,'c':3}
u_dict = {'a': 2, 'b':1, 'c':4}
prob_rank_dict = digest_rank(o_dict, u_dict)
    
# 输出prob_rank_dict = {'b': 2, 'c': 4, 'a': 5}
把所有故障index对应的digest特征都用上面的方法算一遍，整理成一个dataframe： 
例：假设第1个故障对应的cmdb被分了3个digest，分别为0，1，2，根因所在的digest是digest1.
index
change_start_max
change_start_min
……
label（只有0,1两个值）
1
0(表示digest0中没有根因)
1
1(表示digest1中有根因)
1
0(表示digest2中没有根因)
上面的dataframe根据index(重要！！)进行训练集和测试集划分，对训练集进行k折交叉验证，用logistic模型，最后看测试集效果。加上kpi的rank之后再看一遍效果。
    

# 测试函数
# 6个cmdb,6个 KPI
change_start_arr = np.random.randint(1,5,size=[6,6])
o_arr =  np.random.random(((6,6)))
u_arr =  np.random.random(((6,6)))
digests = np.array([0,1,2,3,1,2])
df = digest_features(change_start_arr, o_arr, u_arr, digests)


def digest_rank(o_dict, u_dict):
    '''
    dict_kpi：把kpi和一个数字对应，后面无论是df还是array中的kpi的顺序全部按照字典标号决定。
    参数:
    o_dict: dict,某一个digest中的所有kpi的overflow概率，shape = 所有kpi的个数
            {itemid1:o1, itemid2:o2}
    u_dict: dict,某一个digest中的kpi的underflow概率
    返回：
    prob_rank_dict: dict, v = max{o,u}的值排序 shape = 所有kpi的个数
            {itemid2:v2, itemid1:v1}
    '''
    prob_rank_dict = {}
    for key, value in o_dict.items():
        prob_rank_dict[key] = max(value, u_dict[key])
    prob_rank_dict = dict(sorted(prob_rank_dict.items(), key=lambda d: d[1]))
    return prob_rank_dict
        
    
# 测试函数
o_dict = {'a': 5, 'b':2,'c':3}
u_dict = {'a': 2, 'b':1, 'c':4}
prob_rank_dict = digest_rank(o_dict, u_dict)
    
    