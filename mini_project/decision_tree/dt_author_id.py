#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools_mk/'))
sys.path.append(tools_path)

from emails_Process_mk import preprocess_mk


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess_mk()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time() - t0, 3), "s")
pred = clf.predict(features_test)
t1 = time()
acc = accuracy_score(labels_test, pred)
print("Prediction time :", round(time() - t1, 3), "s" )

print("Accuracy :", round(acc, 3))

print(f"Number of feaures data training: {len(features_train[0])}")


#########################################################
### your code goes here ###


#########################################################


