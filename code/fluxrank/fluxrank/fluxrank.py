import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append(r'E:\AiOps\AIOps_datasets_2020\code\fluxrank\fluxrank\\')
import digest_distillation as dd
import change_qualification as cq
import digest_ranking as dr


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


###################################### 读文件，一次性全部读完 ##########################
filedir = r'E:\AiOps\AIOps_datasets_2020\\'

print('读取故障记录文件')
# 读故障结果文件
fault_filename = '故障整理all.xlsx'
fault = pd.read_excel(filedir+fault_filename)
# start_time转成时间戳
fault['start_time'] = fault.apply(lambda x: datetime_timestamp(str(x['start_time'])), axis=1)
print('读取故障记录文件完成！！')

print('读取所有kpi数据文件')
# 读所有的数据(os,db,docker分开)
file = ['2020_05_26',
        '2020_05_27',
        '2020_05_28',
        '2020_05_29',
        '2020_05_30',
        '2020_05_31']

file_name = ['db_oracle_11g.csv', 'dcos_docker.csv', 'os_linux.csv']

object_name = ['db', 'docker', 'os']

df_list = []
for fname in file_name:
    print("正在读取"+fname)
    df = pd.read_csv(filedir+file[0]+'\平台指标\\'+fname)
    for f in tqdm(range(1,len(file))):
        df = df.append(pd.read_csv(filedir+file[f]+'\平台指标\\'+fname))
    df_list.append(df)
        
print('读取KPI数据文件完成！！')   
    
     
########################## 建立字典：db,docker,os各一个字典, ########################
######################里面的kpi按名字和名字对应的kpi个数各建立2个list #####################
# 3种网元(cmdb)的list
cmdb_list = []
for j in range(3):
    temp_list = list(set(df_list[j]['cmdb_id']))
    cmdb_list.append(temp_list)

# 3种网元字典构成的list
cmdb2idx_list = []
for j in range(3):
    cmdb2idx = {d:i for i, d in enumerate(cmdb_list[j])}
    cmdb2idx_list.append(cmdb2idx)

print('建立网元字典完成！！')
    
print('正在建立kpi构成的list，统计每个name对应的kpi个数')    
# 3种网元的kpi构成的list
cmdb_name_list = []
name_cnt_list = [] # 每个name在所有cmdb中最多出现多少次
for j in range(3):
    # 每种网元中的名称
    print("正在处理"+object_name[j])
    name_list = list(set(df_list[j]['name']))
    cmdb_name_list.append(name_list)
    name_cnt = list(np.zeros(len(name_list)))
    for n in tqdm(range(len(name_list))):
        name = name_list[n]
#        print(name)
        df_name = df_list[j][df_list[j]['name'] == name]
        temp = df_name.groupby(['cmdb_id','itemid']).count().reset_index()
        name_cnt[n] = max(temp.groupby('cmdb_id').count()['itemid'])
    name_cnt_list.append(name_cnt)

print('建立KPI list完成！！')
#################################建立所有kpi的digest和max(pu,po)的df##############
#################################建立所有digest和label的df##############

# 截取时间段：故障发生时间+[-30,10]min
timerange = np.array([-30,10])*60*1000 # 转化成ms

# 对每个fault index 计算 每一个kpi的change start time, po, pu

df_all = pd.DataFrame()
df_reg = pd.DataFrame()

# 对每个fault index
for i in range(79, len(fault)):
    fault_info = fault.iloc[i,:]
    index = fault_info['index']
    fault_timestamp = fault_info['start_time']
    #对每种容器类型
    for j in range(3):
        print('故障'+str(index)+':正在对网元类型'+object_name[j]+'进行特征提取和聚类')
        # 对每个网元
        # 用于聚类
        cmdb_features = []
        # 用于回归
        all_features = [] # cmdb_id, kpi_name_id, kpi_item_id change_start, o , u
        bar = tqdm(range(len(cmdb_list[j])))
        for idx in bar:
            cmdb = cmdb_list[j][idx]
            bar.set_description(f'故障'+str(index)+':正在计算'+cmdb+'特征')
            df_cmdb = df_list[j][df_list[j]['cmdb_id']==cmdb]
            # 截取时间窗
            df_cmdb_fault = df_cmdb.loc[(df_cmdb['timestamp'] >= fault_timestamp+timerange[0])&
                                    (df_cmdb['timestamp'] <= fault_timestamp+timerange[1])]
            #用于聚类
            kpi_features = []
            for n in range(len(cmdb_name_list[j])):
                name = cmdb_name_list[j][n]
                # 提取所有name关联的itemid
                df_name = df_cmdb_fault[df_cmdb_fault['name'] == name]
                
                # 提取里面所有的kpi的itemid
                kpi_list = list(set(df_name['itemid']))
                            
                # 对每个kpi，不管存不存在;目的：保证features维数一样
                for k in range(name_cnt_list[j][n]):
                    if k <= len(kpi_list)-1:
                        itemid = kpi_list[k]
                        df_kpi = df_name[df_name['itemid']==itemid].sort_values(by = 'timestamp', ascending = True)
                
                        # 计算三项特征
#                        print(df_kpi)
                        feature_list = cq.change_start_timestamp(np.array(df_kpi['timestamp']), 
                                                                 np.array(df_kpi['value']), fault_timestamp)
                        # 用于聚类
                        kpi_features = kpi_features+feature_list[1:]
                        # 用于回归
                        all_features.append([index, j, idx, n, k, itemid]+feature_list)
                        
                    else:
                        kpi_features = kpi_features+[0,0]
                        all_features.append([index, j, idx, n, k, 0, 0, 0, 0])
            
            cmdb_features.append(kpi_features)
            
        cmdb_features = np.array(cmdb_features)
#        print(cmdb_features)
        df_all_features = pd.DataFrame(all_features, columns = ['index', 'cmdb_type', 'cmdb', 'kpi_name', 'kpi_item', 'itemid', 'change_start', 'po', 'pu'])
        
        # cmdb聚类
        digests = dd.digest_distillation(cmdb_features)
        
        # 整理digest特征，为最后的训练测试集做准备
        df_digest_features = dr.digest_features(i, j, df_all_features, digests)
        
        # 整理label特征，为最后的训练测试集做准备
        
        df_label_digest = dr.label_digest(object_name, df_digest_features, fault_info, cmdb2idx_list, digests)
        
        # 整理所有cmdb和kpi的所属digest
        df_all_features['digest'] = df_all_features.apply(lambda x: digests[int(x['cmdb'])], axis=1)
        df_all_features['pmax'] = df_all_features.apply(lambda x: max(x['po'], x['pu']), axis=1)
        
        df_all = df_all.append(df_all_features)
        df_reg = df_reg.append(df_label_digest)

    

    
    

   
