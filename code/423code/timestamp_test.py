# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:21:16 2021

@author: CZY
"""

import pandas as pd

from granger import find_timeseries_interval

if __name__ == '__main__':
    
    '''
    db_filedir_23 = r"E:\AiOps\AIOps_datasets_2020\2020_04_23\平台指标\db_oracle_11g.csv"  
    df_db_23 = pd.read_csv(db_filedir_23)   
    '''
   
    os_filedir = r"E:\AiOps\AIOps_datasets_2020\2020_04_23\平台指标\os_linux.csv"
    df_os = pd.read_csv(os_filedir)
    
    cmdb_id ='os_021'
    
    df_os = df_os[df_os['cmdb_id']=='os_021'] 
    df_os = df_os.reset_index(drop = True)
    
    name_list = list(set(df_os['name']))
    len_name = len(name_list)
    df_gap = pd.DataFrame()
    
    for k in range(len_name):
        df_temp = find_timeseries_interval(df_os, [cmdb_id,name_list[k]], coltype='cmdb_name')
        df_temp = df_temp.reset_index(drop = True)
        interval_list = []
        for i in range(len(df_temp)-1):
            interval_list.append(df_temp['timestamp'][i+1]-df_temp['timestamp'][i])
        temp = {'name': name_list[k], 'intervals': interval_list} 
        df_gap = df_gap.append(temp,ignore_index=True)
        
            
        
    