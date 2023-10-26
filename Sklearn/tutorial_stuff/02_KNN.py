# Infos about KNN
# kann supervied und unsupervised verwendet werden
# empfindlich gegenüber Ausreißern, vergleichsweise rechenintensiv
# n_neigbours(k) muss sinnvoll gewählt werden:
# Im Trainig merkt sich der Algo alle vorhandenen Datenpunkte, 
# um bei der Vorhersage die 'k' nächsten Nachbarn zu finden um eine Klasse zuzuordnen




import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('./Sklearn/tutorial_stuff/tutorial_data/CAR_EVALUATION.csv')


X = data[['buying', 'maint', 'safety']].values
y = data[['Target']]

# Converting data
## Method 1  (LabelEncoder)
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

## Method 2 (Mapping)
label_maping = {
    'unacc':0, 'acc':1, 'good':2, 'vgood':3
}
y['Target'] = y['Target'].map(label_maping)
y = np.array(y)

# create Model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)

print(f'Predictions: {predictions}')
print(f'Accuracy: {accuracy}')

a = 1727
print("actual value ", y[a])
print("predicted value", knn.predict(X)[a])




