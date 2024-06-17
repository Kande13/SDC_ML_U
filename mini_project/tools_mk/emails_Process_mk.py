import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


def preprocess_mk():

    tools_path_mk = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools_mk/'))
    print(tools_path_mk)
    authors_file = f"{tools_path_mk}/email_authors.pkl"
    words_file = f"{tools_path_mk}/word_data.pkl"


    with open(authors_file, "rb") as af:
        authors = joblib.load(af)

    with open(words_file, "rb") as wf:
        words = joblib.load(wf)


    features_train, features_test, labels_train, labels_test = train_test_split(words, authors, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    # Sélectionner les caractéristiques les plus importantes
    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    print("No. of Chris training emails : ", sum(labels_train))
    print("No. of Sara training emails : ", len(labels_train)-sum(labels_train))

    return features_train_transformed, features_test_transformed, labels_train, labels_test


if __name__ == "__main__":
    print()
    #preprocess_mk()

