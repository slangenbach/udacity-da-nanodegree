import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn import tree

# construct classifier
clf = tree.DecisionTreeClassifier(min_samples_split=2)
clf2 = tree.DecisionTreeClassifier(min_samples_split=50)

# fit classifier
clf.fit(features_train, labels_train)
clf2.fit(features_train, labels_train)

# predict
pred = clf.predict(features_test)
pred2 = clf2.predict(features_test)

# print accuracy score
from sklearn.metrics import accuracy_score
acc_min_samples_split_2 = accuracy_score(pred, labels_test)
acc_min_samples_split_50 = accuracy_score(pred2, labels_test)


### be sure to compute the accuracy on the test set
def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

