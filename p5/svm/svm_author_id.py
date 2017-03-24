#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
# import SVM algorithm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# create classifier
clf = SVC(kernel="rbf", C=10000)

# reduce training set
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

# fit classifier
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

# predict authors
t0 = time()
pred = clf.predict(features_test)
print("prediction time: ", round(time()-t0, 3), "s")

# print accuracy
print("accuracy: ", accuracy_score(labels_test, pred))

# print confusion matrix
print(confusion_matrix(labels_test, pred))

# print accuracy for items 10, 26, 50
#print(labels_test[10], pred[10], labels_test[26], pred[26], labels_test[50], pred[50])

#########################################################