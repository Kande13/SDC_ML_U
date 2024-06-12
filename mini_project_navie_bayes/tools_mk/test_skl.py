from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

# Exemple de données
documents = ["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"]
labels = [0, 1, 1, 0]  # Juste pour séparer en train/test

# Diviser les données en ensembles d'entraînement et de test
features_train, features_test, labels_train, labels_test = train_test_split(documents, labels, test_size=0.5, random_state=42)

# Créer un TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

# Transformer les données d'entraînement
features_train_transformed = vectorizer.fit_transform(features_train)

# Transformer les données de test
features_test_transformed = vectorizer.transform(features_test)

# Sélectionner les caractéristiques les plus importantes
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)

# Vérifier si des caractéristiques ont été sélectionnées
if selector.get_support().sum() == 0:
    print("Aucune caractéristique n'a été sélectionnée. Ajustez les paramètres de sélection.")
else:
    # Transformer les données d'entraînement et de test
    features_train_transformed = selector.transform(features_train_transformed)
    features_test_transformed = selector.transform(features_test_transformed)

    # Si les transformations sont déjà des numpy.ndarray, pas besoin de .toarray()
    # Afficher les matrices de caractéristiques après sélection
    print("Matrice de caractéristiques d'entraînement après sélection :")
    print(features_train_transformed)
    print("Matrice de caractéristiques de test après sélection :")
    print(features_test_transformed)
