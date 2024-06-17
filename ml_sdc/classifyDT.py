def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
      
    return clf.fit(features_train, labels_train)
