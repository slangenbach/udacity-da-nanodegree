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

# select feature based on intuition
#features_list = ["poi", "bonus", "exercised_stock_options", "from_poi_to_this_person", "long_term_incentive", "shared_receipt_with_poi", "total_stock_value"]

# start with all features and do selection via statistcal technique
features_list = ["poi", 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                'shared_receipt_with_poi']

### Task 2: Remove outliers

# remove TOTAL and travel agency lines
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

# remove outliers identified during analysis
#data_dict.pop("LAY KENNETH L", 0)
#data_dict.pop("SKILLING JEFFREY K", 0)


### Task 3: Create new feature(s)
# see jupyter notebook

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from numpy import arange
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# Preprocess features
features = Imputer(strategy="median").fit_transform(features)

# base clf
from sklearn.naive_bayes import GaussianNB

pipe1 = Pipeline([
    ("clf", GaussianNB()) # provided as base clf
])

# clf selection with grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Define scalers for pipe
ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler()

# Define features selection for pipe
kbest = SelectKBest(k=9)
pbest = SelectPercentile(percentile=20)
pca = PCA(n_components=5)

# Define clfs for pipe nb, svc, knn, rf and ada
nb = GaussianNB()
svc = SVC()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()

# pipe for gs1
pipe_gs = Pipeline([
    ("scaler", None),
    ("feature_selection", None),
    ("clf", nb)
])

# set params for grid search
params = {
    "scaler": [ss, mms, rs],
    "feature_selection": [kbest, pbest, pca],
    "clf": [nb, svc, knn, rf, ada]
}

# define cv for grid search
cv = StratifiedShuffleSplit(n_splits=10, random_state=42)

gs = GridSearchCV(pipe_gs, param_grid=params, cv=cv, scoring="f1_micro", n_jobs=-1) #f1
#gs.fit(features, labels)
#print(gs.best_params_)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# define hyperparameters to be tuned

# params for ada
# params2 = {
#     "ada__n_estimators": range(10, 100+1, 5),
#     "ada__learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
# }

# params for rf
params2 = {
    "rf__n_estimators": range(1, 50+1, 10),
    "rf__max_depth": range(2, 20+1, 5),
    "rf__min_samples_split": [2, 4, 8, 10],
    #"rf__min_samples_leaf": range(2, 10+1, 2),
    "rf__max_leaf_nodes": range(2, 10+1, 2),
}

# pipe for gs2
pipe_gs2 = Pipeline([
    ("scaler", ss), # rs determined by gs1
    ("feature_selection", pbest), # kbest determined by gs1
    #("ada", ada) # ada determined by gs1
    ("rf", rf)
])

# hyperparameter tuning via grid search
gs2 = GridSearchCV(pipe_gs2, param_grid=params2, cv=cv, scoring="f1_micro", n_jobs=-1) # f1
#gs2.set_params(rf__verbose=1, rf__class_weight= {0: 0.3, 1: 0.7})
#gs2.fit(features, labels)
#print(gs2.best_params_)

# pipe for prediction
pipe2 = Pipeline([
    ("scaler", None), # None, because scaling not necessary with trees
    ("feautre_selection", pbest),
    #("clf", ada)
     ("clf", rf) # ada
])

# set params found by gs2
#pipe2.set_params(clf__n_estimators=21, clf__learning_rate=0.8) # n_estimators = 21, learning_rate = 0.8
pipe2.set_params(clf__min_samples_split=4, clf__max_depth=12, clf__n_estimators=21,
                 clf__class_weight={0.0: 0.45, 1.0: 0.55})

# train-test split for prediction
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# fit clf
clf = pipe2
#clf.fit(features, labels)
clf.fit(features_train, labels_train)

# predict results
pred = clf.predict(features_test)

# print scores
from sklearn.metrics import classification_report

#print classification_report(labels, pred) # base clf f1: 0.41, tuned ada f1: 0.77, tuned rf f1: 0.81 (all for poi=1.0)
print(classification_report(labels_test, pred)) # rf: (0.29 poi=1.0, 0.82 total), ada: (0.29 for poi=1.0, 0.77 total)

# custom score function
def custom_score(est=clf, features=features_test, labels=labels_test):
    """
    Take scikit-learn estimator, features and labels, predict results and return score
    :param est: scikit-learn estimator
    :param features:  input features
    :param labels: input labels
    :return: scikit-learn f1 score if precision score and recall > 0.3 else 0
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    pred = est.predict(features)
    pre = precision_score(labels, pred, average='micro')
    rec = recall_score(labels, pred, average='micro')

    if pre > 0.3 and rec > 0.3:
        return f1_score(labels, pred, average='macro')
    else:
        return 0

#print(custom_score())

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)

# run clf against tester
# ToDo: https://discussions.udacity.com/t/trying-to-hit-over-0-3/196167/27
from tester import test_classifier
test_classifier(clf, my_dataset, features_list)
