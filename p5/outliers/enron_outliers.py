#!/usr/bin/python
from __future__ import print_function

import numpy as np
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

# remove TOTAL line from dict
data_dict.pop("TOTAL", 0)

# prepare data
data = featureFormat(data_dict, features)


### your code below

# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     matplotlib.pyplot.scatter( salary, bonus )
#
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

# print information for TOTAL key
#print(data_dict["TOTAL"])

# print information for remaining outliers
for k,v in data_dict.iteritems():
    if v["salary"] != "NaN" and v["bonus"] != "NaN":
        if v["salary"] >= 1000000 and v["bonus"] >= 5000000:
            print(k, v["salary"], v["bonus"])

