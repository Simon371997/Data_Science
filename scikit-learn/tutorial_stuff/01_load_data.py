import sklearn
from sklearn import datasets
from sklearn.datasets import fetch_openml

iris = datasets.load_iris()
print(iris.feature_names)
print(iris.data)
print(iris.target_names)

mice = fetch_openml(name='miceprotein', version=4)
print(mice.details)