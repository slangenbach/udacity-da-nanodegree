#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
# enable python 3 style printing
from __future__ import print_function

# setup environment
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################


# import NB algorithm
from sklearn.naive_bayes import GaussianNB

# create classifier
clf = GaussianNB()

# train classifier
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

# predict authors
t0 = time()
pred = clf.predict(features_test)
print("prediction time: ", round(time()-t0, 3), "s")

# print accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, labels_test))


#########################################################

