import numpy

from sklearn.kernel_approximation import RBFSampler

#from nab.detectors.base import AnomalyDetector

import pandas as pd
import numpy as np

path = r'E:\AiOps\AIOps_datasets_2020\2020_04_11\业务指标\esb.csv'
esb = pd.read_csv(path)
avg_time = esb['avg_time']



previousExposeModel = []
decay = 0.01
timestep = 0



"""Initializes RBFSampler for the detector"""
kernel = RBFSampler(gamma=0.5,
                         n_components=20000,
                         random_state=290)


#  def handleRecord(self, inputData):
""" Returns a list [anomalyScore] calculated using a kernel based
similarity method described in the comments below"""

# Transform the input by approximating feature map of a Radial Basis
# Function kernel using Random Kitchen Sinks approximation
inputFeature = kernel.fit_transform(
  numpy.array([[inputData["value"]]]))

# Compute expose model as a weighted sum of new data_origin point's feature
# map and previous data_origin points' kernel embedding. Influence of older data_origin
# points declines with the decay factor.
if timestep == 0:
  exposeModel = inputFeature
else:
  exposeModel = ((decay * inputFeature) + (1 - decay) *
                 previousExposeModel)

# Update previous expose model
previousExposeModel = exposeModel

# Compute anomaly score by calculating similarity of the new data_origin point
# with expose model. The similarity measure, calculated via inner
# product, is the likelihood of data_origin point being normal. Resulting
# anomaly scores are in the range of -0.02 to 1.02.
anomalyScore = numpy.asscalar(1 - numpy.inner(inputFeature, exposeModel))
timestep += 1


