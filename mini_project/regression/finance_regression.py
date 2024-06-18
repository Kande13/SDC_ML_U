#!/usr/bin/python3

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools_mk/"))
sys.path.append(tools_path)

from feature_format import featureFormat, targetFeatureSplit

finalProject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../final_project'))
sys.path.append(finalProject_path)

with open(f"{finalProject_path}/final_project_dataset_modified.pkl", "rb") as fpd:
    dictionary = joblib.load(fpd)

### Fonction pour effectuer la régression et afficher les résultats
def perform_regression(features_list):
    data = featureFormat(dictionary, features_list, remove_any_zeroes=True, sort_keys=f"{tools_path}/python2_lesson06_keys.pkl")
    target, features = targetFeatureSplit(data)

    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
    
    # Conversion des listes en arrays numpy et redimensionnement
    feature_train = np.array(feature_train).reshape(-1, 1)
    feature_test = np.array(feature_test).reshape(-1, 1)
    target_train = np.array(target_train)
    target_test = np.array(target_test)

    # Création et ajustement du modèle de régression sur les données d'entraînement
    reg = LinearRegression()
    reg.fit(feature_train, target_train)

    # Extraction de la pente et de l'interception
    slope_train = reg.coef_[0]
    intercept_train = reg.intercept_
    print(f"Pente (slope) des données d'entraînement: {slope_train}")
    print(f"Interception des données d'entraînement: {intercept_train}")

    # Prédictions sur les données de test
    predictions_train = reg.predict(feature_test)

    # Ajustement du modèle de régression sur les données de test
    reg.fit(feature_test, target_test)
    
    # Extraction de la pente et de l'interception pour les données de test
    slope_test = reg.coef_[0]
    intercept_test = reg.intercept_
    print(f"Pente (slope) des données de test: {slope_test}")
    print(f"Interception des données de test: {intercept_test}")

    # Prédictions sur les données d'entraînement
    predictions_test = reg.predict(feature_train)

    # Tracé du scatterplot avec les points de test et d'entraînement
    for feature, target in zip(feature_test, target_test):
        plt.scatter(feature, target, color="r")
    for feature, target in zip(feature_train, target_train):
        plt.scatter(feature, target, color="b")

    # Labels pour la légende
    plt.scatter(feature_test[0], target_test[0], color="r", label="test")
    plt.scatter(feature_train[0], target_train[0], color="b", label="train")

    # Tracé des lignes de régression
    plt.plot(feature_test, predictions_train, color='green', label='Regression on training data')
    plt.plot(feature_train, predictions_test, color='blue', label='Regression on test data')

    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()
    plt.show()

### Régression du bonus par rapport au salaire
print("Régression du bonus par rapport au salaire:")
perform_regression(["bonus", "salary"])
