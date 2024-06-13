import sys
from time import time
import os


tools_path = f"{os.path.dirname(__file__)}/tools_mk"
sys.path.append(tools_path)

from emails_Process_mk import preprocess_mk

#Prétraiter les données
features_train, features_test, labels_train, labels_test = preprocess_mk()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Créer le classificateur
clf = GaussianNB()

#Entrainer le classificateur
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time() - t0, 3), "s")

# Faire des predictions
t1 = time()
pred = clf.predict(features_test)
print("Prediction Time:", round(time() - t1, 3), "s")

# Calculer et afficher la prédiction
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {round(accuracy, 3)}")







