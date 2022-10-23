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
    clustering = DBSCAN(eps=0.07, min_samples=2, metric='precomputed')
    X = pd.DataFrame(cmdb_features).T
    distance_matrix = X.corr(method='pearson')
    distance_matrix = distance_matrix.fillna(0)
#    print(distance_matrix)
    digests = clustering.fit(abs(distance_matrix)).labels_
    k = max(digests)
    for i in range(len(digests)):
        if digests[i] == -1:
            digests[i] = k+1
            k+=1
    return np.array(digests)



if __name__=="__main__":
    # 测试函数
    cmdb_features =  np.random.random((10,10))
    digests = digest_distillation(cmdb_features)
    print(digests)