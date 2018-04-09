import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, tree

x, y = datasets.load_wine(True)
threshold = int(np.floor(len(x) * .85))
np.random.shuffle(x)
np.random.shuffle(y)
x_train = x[:threshold]
x_test = x[threshold:]
y_train = y[:threshold]
y_test = y[threshold:]

dt = tree.DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)

y_predict = dt.predict(x_test)

score = metrics.f1_score(y_test, y_predict, average='weighted')

def main():
    print('f1_score', score)
