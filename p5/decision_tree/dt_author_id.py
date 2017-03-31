#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
# enable python 3 style printing
from __future__ import print_function


import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
# print number of features used
# print(len(features_train[1]))

from sklearn import tree

# create clf
clf = tree.DecisionTreeClassifier(min_samples_split=40)

# fit clf
clf.fit(features_train, labels_train)

# predict
pred = clf.predict(features_test)

# print accuracy score
from sklearn.metrics import accuracy_score
print("accuracy: ", accuracy_score(labels_test, pred))


#########################################################


