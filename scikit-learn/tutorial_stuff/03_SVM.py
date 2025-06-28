# Infos about SVM
# supervised, Classification
# sehr gut bei vielen Merkmalen
# robust gegenüber Ausreißern
# rechenintensiv

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# Load data
iris = datasets.load_iris()


# Split data
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# model 
model = svm.SVC()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'accuracy: {accuracy}')



