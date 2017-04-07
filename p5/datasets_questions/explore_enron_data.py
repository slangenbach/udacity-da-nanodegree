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
from __future__ import division

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print number of rows in dataset
# print(len(enron_data))
#
# # print number of features in dataset
# print([len(v) for v in enron_data.values()])
#
# # print number of POIs ink pickle file
# cnt = 0
#
# for k,v in enron_data.iteritems():
#     if v["poi"] == 1:
#         cnt += 1
#
# print(cnt)

# print number of POIs in txt file
# with open("../final_project/poi_names.txt", "r") as f:
#
#     cnt = 0
#
#     for l in f.readlines():
#
#         if l.startswith("(y"):
#
#             cnt += 1
#
#     print(cnt)

# print keys in enron dict
# print(enron_data.viewkeys())

# print values of enron dict
# print(enron_data.viewvalues())

# print number of stock w.r.t. James Prentice
# print(enron_data["PRENTICE JAMES"]["total_stock_value"])

# print number of email messages from Wesley Colwell to POIs
# print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# print value of stock options exercised by Jeffrey K Skilling
# print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

# print name and max total_payments value for Lay, Skilling and Fastow
# for k,v in enron_data.iteritems():
#     if k in ["LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"]:
#         print(k,v["total_payments"])

# print number of individuals with quantified salary or known email address:
# cnt_salary = 0
# cnt_email = 0
# cnt_payments = 0
# cnt_payments_poi = 0
#
# for k, v in enron_data.iteritems():
#     if v["salary"] != "NaN":
#         cnt_salary += 1
#     if v["email_address"] != "NaN":
#         cnt_email += 1
#     if v["total_payments"] == "NaN":
#         cnt_payments += 1
#     if v["poi"] == 1 and v["total_payments"] == "NaN":
#         cnt_payments_poi += 1
#
# print("# salary: %d, # email addresses %d, # total_payments %d (in percentage: %f), poi and total_payments %d (%f)" %
#       (cnt_salary, cnt_email, cnt_payments, cnt_payments/146, cnt_payments_poi, cnt_payments_poi/146))