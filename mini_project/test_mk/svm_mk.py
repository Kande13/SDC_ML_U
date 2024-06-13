################ Étape 1 : Importer les Bibliothèques Nécessaires ##############################

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

##################### Étape 2 : Charger et Préparer les Données ################################

# Charger un jeu de données (par exemple, les Iris de Fisher)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Utilisation des deux premières caractéristiques pour simplifier la visualisation
y = iris.target

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

######### Étape 3 : Créer et Entraîner le Modèle SVM #######################################

# Créer le modèle SVM avec un noyau linéaire
model = SVC(kernel='linear')

# Entraîner le modèle
model.fit(X_train, y_train)

########################## Étape 4 : Évaluer le Modèle ########################################

# Prédire les étiquettes pour les données de test
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')


######################## Étape 5 : Visualiser les Résultats  #####################################

# Fonction pour tracer l'hyperplane et les vecteurs de support
def plot_svm_decision_boundary(model, X, y):
    h = .02  # Taille des pas dans la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# Tracer la frontière de décision
plot_svm_decision_boundary(model, X_test, y_test)

