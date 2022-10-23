# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:24:17 2021

@author: CZY
"""


from ESE import ExposeDetector
import pandas as pd
import numpy as np

path = r'E:\AiOps\AIOps_datasets_2020\2020_04_11\业务指标\esb.csv'
esb = pd.read_csv(path)
avg_time = esb['avg_time']

ED = ExposeDetector()

x = ED.handleRecord(avg_time)