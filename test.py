import streamlit as st
import seaborn as sns

st.header("Hi Sanchit here, My first streamlit app")
st.text("Working on IRIS Dataset")

from sklearn.datasets import load_iris
iris= load_iris()
# Store features matrix in X
X= iris.data
#Store target vector in 
y= iris.target

# Names of features/columns in iris dataset
print(iris.feature_names)

print(iris.target_names)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier()

knn.fit(X, y)

#Predicting output of new data
knn.predict([[3.2, 5.4, 4.1, 2.5]]) # Random values


# instantiate the model 
knn = KNeighborsClassifier(n_neighbors=1)# fit the model with data
knn.fit(X, y)# predict the response for new observations
knn.predict([[3, 5, 4, 2]])

# For k = 6

# instantiate the model 
knn = KNeighborsClassifier(n_neighbors=6)# fit the model with data
knn.fit(X, y)# predict the response for new observations
knn.predict([[3, 5, 4, 2]])


#  split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

print(X_train.shape)
print(X_test.shape)


# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# For k = 1

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))


# For k = 6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))


# try K=1 through K=30 and record testing accuracy
k_range = list(range(1, 31))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))



# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')





from sklearn.datasets import load_iris
iris= load_iris()
# Store features matrix in X
X= iris.data
#Store target vector in y
y= iris.target
# Finalizing KNN Classifier after evaluation and choosing best 
# parameter
#Importing KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=12)
# train the model with X and y (not X_train and y_train)
knn_clf=knn.fit(X, y)
# Saving knn_clf
import joblib
# Save the model as a pickle in a file
joblib.dump(knn_clf, "Knn_Classifier.pkl")








