#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import os   
import sys
from time import time


tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools_mk/'))
sys.path.append(tools_path)

from emails_Process_mk import preprocess_mk


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess_mk()


#########################################################
### your code goes here ###

from  sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(kernel='rbf', C=10000)

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train, labels_train)

print("Training time:", round(time() - t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("Prediction time:", round(time() - t1, 3), "s")

print("Chris:", sum(pred))

acc = accuracy_score(labels_test, pred)

print("Accuracy:", round(acc, 3))

# Extract predictions for specific elements
pred_10 = pred[10]
pred_26 = pred[26]
pred_50 = pred[50]
pred_100 = pred[100]

print("Prediction for element 10:", pred_10)
print("Prediction for element 26:", pred_26)
print("Prediction for element 50:", pred_50)
print("Prediction for element 100:", pred_100)

# Count the number of emails predicted to be from Chris (class 1)


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''


#########################################################
