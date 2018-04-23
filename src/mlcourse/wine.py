from pprint import pprint
from sklearn import datasets, metrics, model_selection, tree
import graphviz
import matplotlib.pyplot as plt
import numpy as np

# load and shuffle dataset
x, y = datasets.load_wine(True)
zipped = list(zip(x, y))
np.random.shuffle(zipped)
x, y = zip(*zipped)

# slice samples into test and train
threshold = int(np.floor(len(x) * .85))
x_train = x[:threshold]
x_test = x[threshold:]
y_train = y[:threshold]
y_test = y[threshold:]

# create tree and inital fitting
dt = tree.DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)

# specify paramter search space
random_grid = dict(
    criterion=['gini', 'entropy'],  # quality of a split
    splitter=['best', 'random'],  # split strategy
    max_depth=[5, 10, 20],  # maximum depth of the tree
    min_samples_split=[0.1, 0.6, 1.0],  # min samples to split node
    min_samples_leaf=[0.1, 0.3, 0.5],  # min leaf nodes required
    min_weight_fraction_leaf=[0.0, 0.25, 0.5],  # minimum weighted fraction
    max_features=['auto', 'sqrt', 'log2', None],  # features considered
    max_leaf_nodes=[10, 25, 50],  # max leaf nodes
    presort=[True, False])  # presort data before fitting

# perform exhaustive paramter search
selector = model_selection.GridSearchCV(
    estimator=dt,
    param_grid=random_grid,
    error_score=0,
    refit=True,
    verbose=1,
    n_jobs=4)
selector.fit(x_train, y_train)

# output optimal parameter values
print('best_socre', selector.best_score_)
pprint(selector.best_params_)

# calculate model accuracy
y_predict = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_predict)
print('accuracy', accuracy)


def main():
    print(__name__)
