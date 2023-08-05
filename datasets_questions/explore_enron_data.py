#!/usr/bin/python3

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

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))
print(len(enron_data))
#print(len(enron_data['METTS MARK']))
#print(enron_data)

count = 0
for key, values in enron_data.items():
    # Process each key and its corresponding values
    if enron_data[key]["poi"]==1:
        count +=1

print (count)


count = 0
for key, value in enron_data.items():
    if enron_data[key]["poi"]==1 and enron_data[key]['total_payments'] == 'NaN':
        count +=1
print(count)

print((count*100)/len(enron_data))
    
