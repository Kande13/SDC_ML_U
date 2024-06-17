import os
import sys
from time import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools_mk'))
sys.path.append(tools_path)

from emails_Process_mk import preprocess_mk

features_train, features_test, labels_train, labels_test = preprocess_mk()

neigh = KNeighborsClassifier(n_neighbors=5)

t0 = time()
neigh.fit(features_train, labels_train)
print(f"Trainning time: {round(time() - t0, 3)}")

t1 = time()
pred = neigh.predict(features_test)
print(f"Prediction time : {round(time() - t1, 3)}")

acc = accuracy_score(labels_test, pred)
print(f"Accuracy : {round(acc, 3)}")

