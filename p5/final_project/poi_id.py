#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "exercised_stock_options", "total_stock_value", "bonus"]

### Task 2: Remove outliers # ToDo: get rid of outliers
# remove TOTAL line
data_dict.pop("TOTAL", 0)

# remove NaN values
for k, v in data_dict.iteritems():
    for k2, v2 in v.iteritems():
        if v2 == "NaN":
            v[k2] = 0

# plot features # ToDo evaluate feature importance

### Task 3: Create new feature(s)
for k, v in data_dict.iteritems():
        v["performance_compensation"] = v["bonus"] + v["exercised_stock_options"] + v["long_term_incentive"]

        # prevent zero division error
        if v["to_messages"] == 0:
            v["poi_comm_index"] = 0
        else:
            v["poi_comm_index"] = float(v["from_this_person_to_poi"] / v["to_messages"])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers ToDo implement PCA?
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit # Todo Implement
from sklearn.model_selection import train_test_split

# scale features
#features = StandardScaler().fit_transform(features)
features = MinMaxScaler().fit_transform(features)

# perform train-test split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,test_size=0.3, random_state=42)

# base clf
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

pipe1 = Pipeline([
    ("nb", GaussianNB()) # provided as base clf
])



# clf with feature selection
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
from sklearn.svm import LinearSVC


pipe2a = Pipeline([
    ("feautre_selection", SelectFromModel(LinearSVC())),
    ("clf", GaussianNB())
])

# clf with feature selection and tree
from sklearn.ensemble import RandomForestClassifier

pipe2a = Pipeline([
    ("feautre_selection", SelectKBest(k=4)),
    ("clf", RandomForestClassifier())
])

#
from sklearn.linear_model import LogisticRegressionCV
cv = StratifiedShuffleSplit(n_splits=10)
pipe2c = Pipeline([
    ("log_reg", LogisticRegressionCV(cv=cv, solver="liblinear")) # provided as base clf
])


### Task 5: Tune your classifier to achieve better than .3 precision and recall ToDo implement gridsearch
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.model_selection import GridSearchCV
#clf = RandomForestClassifier()
#params = ""

#gs = GridSearchCV(clf, params=params, )


# Example starting point. Try investigating other evaluation techniques!
# fit clf
clf = pipe1
#clf.fit(features_train, labels_train)
clf.fit(features, labels)

# predict results
#pred = clf.predict(features_test)
pred = clf.predict(features)

# print precision and recall score
from sklearn.metrics import precision_score, recall_score, classification_report
#print "precision: %f" % precision_score(pred, labels_test) # base score: 0.4
#print "recall: %f" % recall_score(pred, labels_test) # base score: 0.4
#print classification_report(labels_test, pred)
print classification_report(labels, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#dump_classifier_and_data(clf, my_dataset, features_list)