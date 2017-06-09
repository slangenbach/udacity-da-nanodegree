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

# strategy 1: select feature based on intuition
#features_list = ["poi", "bonus", "exercised_stock_options", "from_poi_to_this_person", "long_term_incentive", "shared_receipt_with_poi", "total_stock_value"]

# strategy 2: start with all features and do selection via feature selection technique
features_list = ["poi", 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                'shared_receipt_with_poi']

### Task 2: Remove outliers

# remove TOTAL and travel agency lines
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# remove observation with all NaN values
data_dict.pop("LOCKHART EUGENE E", 0)

### Task 3: Create new feature(s)
def add_own_features(data_dict=data_dict, features_list=features_list):
    """
    Take dict, convert it to pandas data frame, replace NaN values with 0 and add two features.
    Append feature labels to feature_list and return data frame as dict
    :param data_dict: input dict
    :return: input dict with two additional features
    """
    import pandas as pd

    # convert dict into df
    df = pd.DataFrame.from_dict(data_dict, orient="index")

    # replace NaN strings with 0
    df.replace(to_replace="NaN", value=0, inplace=True)

    # create performance compensation feature
    df["performance_compensation"] = df["bonus"] + df["exercised_stock_options"] + df["total_stock_value"]

    # create poi communication feature
    df["poi_communication"] = (df["from_poi_to_this_person"] + df["from_this_person_to_poi"]) \
                              / (df["from_messages"] + df["to_messages"] + 1)

    # add new features to feature list
    features_list.append("performance_compensation")
    features_list.append("poi_communication")

    # return df as dict
    return df.to_dict(orient="index")

# include new features in final analysis
data_dict = add_own_features()

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

# pre-process features
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

# define scalers for pipe
ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler()

# define features selection for pipe
kbest = SelectKBest(k=9)
pbest = SelectPercentile(percentile=20)
pca = PCA(n_components=5)

# define clfs for pipe nb, svc, knn, rf and ada
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

gs = GridSearchCV(pipe_gs, param_grid=params, cv=cv, scoring="f1_micro", n_jobs=-1)
#gs.fit(features, labels)
#print(gs.best_params_)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# define hyper-parameters to be tuned
params2 = {
    "rf__n_estimators": range(1, 50+1, 10),
    "rf__max_depth": range(2, 20+1, 5),
    "rf__min_samples_split": [2, 4, 8, 10],
    "rf__max_leaf_nodes": range(2, 10+1, 2),
}

# pipe for gs2
pipe_gs2 = Pipeline([
    ("scaler", None), # rs determined by gs1, but not needed for trees
    ("feature_selection", pbest), # pbest determined by gs1
    #("ada", ada) # ada als alternative to rf
    ("rf", rf)
])

# hyperparameter tuning via grid search
gs2 = GridSearchCV(pipe_gs2, param_grid=params2, cv=cv, scoring="f1_micro", n_jobs=-1)
#gs2.fit(features, labels)
#print(gs2.best_params_)

# pipe for prediction
pipe2 = Pipeline([
    ("scaler", None), # None, because scaling not necessary with trees
    ("feature_selection", pbest),
     ("clf", rf)
])

# set params found by gs2
pipe2.set_params(clf__min_samples_split=4,
                 clf__max_depth=12,
                 clf__n_estimators=21,
                 clf__class_weight={0.0: 0.45, 1.0: 0.55})

# train-test split for prediction
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# fit clf
clf = pipe2
clf.fit(features_train, labels_train)

# predict results
pred = clf.predict(features_test)

# print feature scores
feature_names = list(features_list) # create a copy of features_list
feature_names.pop(0) # remove poi since it is not a feature
feature_scores = clf.named_steps["feature_selection"].scores_
feature_dict = dict(zip(feature_names, feature_scores))
print("feature scores: ", sorted(feature_dict.items(), key=lambda x:x[1], reverse=True)) # ordered by scores

# print clf feature scores and importance
print("number of features used by clf: ", clf.named_steps["clf"].n_features_)
print("feature importance: ", sorted(clf.named_steps["clf"].feature_importances_, reverse=True))

# print classification report
from sklearn.metrics import classification_report
print(classification_report(labels_test, pred)) #rf poi: precision: 0.40, recall: 0.40, f1: 0.40

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)

# run clf against tester # rf: precision: 0.54, recall: 0.31, f1: 0.40
from tester import test_classifier
test_classifier(clf, my_dataset, features_list)