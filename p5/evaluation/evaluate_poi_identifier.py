#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# print number of POIs in test data
#from collections import Counter
#print(Counter(labels_test))

# setup classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

# fit clf
clf.fit(features_train, labels_train)

# predict results
pred = clf.predict(features_test, labels_test)

# print accuracy, recall and precision
from sklearn.metrics import accuracy_score, recall_score, precision_score

print "accurary of decision tree: %f" % accuracy_score(pred, labels_test)
print "recall of decision tree: %f" % recall_score(pred, labels_test)
print "precision of decision tree: %f" % precision_score(pred, labels_test)

