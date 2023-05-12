from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

print(X)
print(y)

## Implementing a KNN classifier

knn = KNeighborsClassifier(n_neighbors=12)

knn_clf = knn.fit(X, y)

import joblib

joblib.dump(knn_clf, "Knn_Classifier.pkl")
