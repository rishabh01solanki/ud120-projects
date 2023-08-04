#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""    
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess
from time import time


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time() # start timer for training
clf.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s") # time taken for training

t0 = time () # prediction time
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s") # time taken for prediction

from sklearn.metrics import accuracy_score
result = accuracy_score(labels_test, pred)
print(result)
##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################