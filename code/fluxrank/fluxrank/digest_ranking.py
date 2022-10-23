import pandas as pd
import numpy as np




def digest_features(index, cmdb_j, df_all_features, digests):
    """
    作用：
    计算每个digest中所有change_start_arr绝对值小于5分钟（暂定）的KPI的
    change_start的 max, min, sum, mean, std; 
    overflow , underflow的max, min, sum,mean 构成每个digest的特征
    
    参数：
    index: int,故障编号
    cmdb_j: 第j个cmdb类型，j=0:db; j=1:docker; j=2:os
    df_all_features: DataFrame, columns包含['cmdb', 'change_start', 'po', 'pu']，这里的cmdb是编号
    digests: digest_distillation函数中的返回格式，表示每个cmdb所属的类
    
    返回：
    df_digest_features: DataFrame，shape = digest个数*16(index+cmdb类型编号+digest名称+feature个数)
    """
    
    digest_cnt = len(list(set(digests)))
    digest_features = []
    for d in range(digest_cnt):
        
        digest_index = list(np.argwhere(digests==d).flatten())
        df_digest = df_all_features.loc[df_all_features['cmdb'].isin(digest_index)]
        df_digest = df_digest.loc[(df_digest['change_start']!=0)|
                                  (df_digest['po']!=0)|
                                  (df_digest['pu']!=0), :]
        df_digest = df_digest.loc[abs(df_digest['change_start'])<=5*60*1000]
        if len(df_digest)==0:
            digest_features.append([index,cmdb_j,d]+[0]*13)
        else:
            digest_features.append([index,cmdb_j,d,
                                    max(df_digest['change_start']),
                                    min(df_digest['change_start']),
                                    sum(df_digest['change_start']),
                                    np.mean(df_digest['change_start']),
                                    np.std(df_digest['change_start']),
                                    max(df_digest['po']), 
                                    min(df_digest['po']),
                                    sum(df_digest['po']), 
                                    np.mean(df_digest['po']), 
                                    max(df_digest['pu']), 
                                    min(df_digest['pu']), 
                                    sum(df_digest['pu']), 
                                    np.mean(df_digest['pu'])])
        
    df_digest_features = pd.DataFrame(digest_features)
    df_digest_features.columns = ['index', 'cmdb_type', 'digest', 
                                  'start_max', 'start_min', 'start_sum', 'start_mean', 'start_std', 
                                  'po_max', 'po_min', 'po_sum', 'po_mean',
                                  'pu_max', 'pu_min', 'pu_sum', 'pu_mean']
    return df_digest_features
    



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


def label_digest(object_name, df_digest_features, fault_info, cmdb2idx_list, digests):
    """
    作用：
    查找真实故障编号对应故障根因所属的类别，作为logistic模型训练的标签
    参数：
    object_name: list, ['db', 'docker', 'os']
    df_digest_features: DataFrame，针对1个故障id，1个cmdb类型(db,docker,os)，提取出来的digest特征
    fault_info: series，1个故障id对应的真实故障信息
    cmdb2idx_list: 存储了所有cmdb和标签的对应信息字典list，长度为3，表示了db,docker,os
    digests: 对于1个故障id，1个cmdb类型的聚类结果
    返回：
    df_digest_lab: 包含故障编号和所有digest的dataframe，并判断该故障编号下根因是否在这个digest中，
                   列名包括特征和['index', 'digest','label']
    """
   # 先全部填0
    df_digest_label = df_digest_features.copy(deep=True)
    df_digest_label['label'] = 0
    
    # 提取故障真实根因
    fault_object = fault_info['object']
    fault_object_idx = np.argwhere(np.array(object_name) == fault_object)[0][0]
    fault_cmdb = fault_info['name']
#    print(fault_cmdb)
    fault_cmdb_idx = cmdb2idx_list[fault_object_idx].get(fault_cmdb)  
#    print(fault_cmdb_idx)
    
    # 如果所在的cmdb类型相同
    if fault_object_idx == list(set(df_digest_label['cmdb_type']))[0]:
        # 根因所在的digest
        fault_digest = digests[fault_cmdb_idx]
        df_digest_label.loc[fault_digest,'label'] = 1
        
    return df_digest_label
        
  
if __name__=='__main__':

    #测试函数label_digest
    
    filedir = r'E:\My source\02   Project\2   Anormaly Detection and Root Cause Location\2   AIOps Challenge\AIOps挑战赛数据'
    filename = '\故障整理all.xlsx'
    df_fault = pd.read_excel(filedir+filename).iloc[:2, :] # temp
    df_cmdb = df_fault[['index', 'name']]
    df_cmdb.columns = ['index', 'cmdb']
    
    dict_cmdb = {'docker_003':0, 'docker_002':1, 'docker_001':2}
    digests = {1:np.array([0,1,1]), 2:np.array([1,2,0])}
    
    label_digest(df_cmdb, digests, dict_cmdb)
    
    
    # 测试函数digest_features
    cmdb = [0,1,2,3,4,5]*3
    features_arr = np.random.random(size=[18,3])# 6个cmdb,3个 KPI
    df = pd.DataFrame(features_arr, columns = ['change_start', 'po', 'pu'])
    df['cmdb'] = cmdb
    index = 0
    cmdb_type = 2
    digests = np.array([0,1,2,3,1,2])
    df_digest = digest_features(index, cmdb_type, df, digests)
    
    
    
    # 测试函数digest_rank
    o_dict = {'a': 5, 'b':2,'c':3}
    u_dict = {'a': 2, 'b':1, 'c':4}
    prob_rank_dict = digest_rank(o_dict, u_dict)
        
    
    # 输出prob_rank_dict = {'b': 2, 'c': 4, 'a': 5}




