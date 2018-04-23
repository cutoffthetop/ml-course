import numpy as np

import graphviz
import matplotlib.pyplot as plt
from sklearn import datasets, tree

iris = datasets.load_iris()


def tree():
    x = iris.data
    y = iris.target
    dt = tree.DecisionTreeClassifier()
    return dt.fit(x, y)


def graph():
    dt_data = tree.export_graphviz(
        tree(),
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dt_data)
    graph.render('output')


def predict(obj):
    p = tree().predict([obj])
    print('this is ', p)


def main():
    predict([4, 3, 1, 0])
