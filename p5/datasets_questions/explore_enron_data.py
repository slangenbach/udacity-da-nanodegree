#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
# python 3 style printing
from __future__ import print_function

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print number of rows in dataset
print([len(enron_data)

# print number of features in dataset
print([len(v) for v in enron_data.values()])

