import os
import sys
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools_mk')))

from emails_Process_mk import preprocess_mk

features_train, features_test, labels_train, labels_test = preprocess_mk()

clf = RandomForestClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print(f"Trainning time : {round(time() - t0, 3)}")

t1 = time()
pred = clf.predict(features_test)
print(f"Prediction time :", round(time() - t1, 3))

acc = accuracy_score(labels_test, pred)
print(f"Accuracy : {round(acc, 3)}")
