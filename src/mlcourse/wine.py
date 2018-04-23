import numpy as np
import graphviz
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, tree, model_selection

wine = datasets.load_wine()

x = wine.data
y = wine.target
zipped = list(zip(x, y))
np.random.shuffle(zipped)
x, y = zip(*zipped)
threshold = int(np.floor(len(x) * .85))
x_train = x[:threshold]
x_test = x[threshold:]
y_train = y[:threshold]
y_test = y[threshold:]

dt = tree.DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)

random_grid = dict(
    criterion=['gini', 'entropy'],  # quality of a split
    splitter=['best', 'random'],  # split strategy
    max_depth=[2, 5, 10, 20],  # maximum depth of the tree
    min_samples_split=[2, 0.2, 0.6, 1.0],  # min samples to split node
    min_samples_leaf=[1, 0.1, 0.3, 0.5],  # min leaf nodes required
    min_weight_fraction_leaf=[0.0, 0.25, 0.5],  # minimum weighted fraction
    max_features=['auto', 'sqrt', 'log2', None],  # features considered
    max_leaf_nodes=[None, 2, 5, 10],  # max leaf nodes
    presort=[True, False])  # presort data before fitting
dt_random = model_selection.GridSearchCV(
    estimator=dt,
    param_grid=random_grid,
    error_score=0,
    verbose=1,
    n_jobs=4)
dt_random.fit(x_train, y_train)
pprint(dt_random.best_params_)

y_predict = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_predict)
print('accuracy', accuracy)


def main():
    print(__name__)
