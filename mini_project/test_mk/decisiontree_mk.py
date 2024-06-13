#Importer les Bibliothèques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Charger et Préparer le Jeu de Données

from sklearn.datasets import load_iris

# Charger le jeu de données
iris = load_iris()
X = iris.data
y = iris.target

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Entraîner le Modèle d'Arbre de Décision

# Créer un objet classificateur d'arbre de décision
clf = DecisionTreeClassifier()

# Entraîner le classificateur d'arbre de décision
clf = clf.fit(X_train, y_train)

# Prédire et Évaluer le Modèle

# Prédire la réponse pour le jeu de test
y_pred = clf.predict(X_test)

# Précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision: {accuracy * 100:.2f}%")

############### Visualiser l'Arbre de Décision ###################

# Tracer l'arbre de décision
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


############# Exemple d'Élagage #######################
# Créer un classificateur d'arbre de décision avec élagage
clf_pruned = DecisionTreeClassifier(max_depth=3)

# Entraîner le classificateur d'arbre de décision
clf_pruned = clf_pruned.fit(X_train, y_train)

# Prédire la réponse pour le jeu de test
y_pred_pruned = clf_pruned.predict(X_test)

# Précision du modèle
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Précision de l'Arbre Élagué: {accuracy_pruned * 100:.2f}%")

# Importance des Caractéristiques

# Obtenir l'importance des caractéristiques
importances = clf.feature_importances_
feature_importance = pd.DataFrame({'Caractéristique': iris.feature_names, 'Importance': importances})
print(feature_importance)
