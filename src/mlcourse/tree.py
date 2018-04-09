import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn import datasets, tree

iris = datasets.load_iris()

x = iris.data
y = iris.target

dt = tree.DecisionTreeClassifier()
dt = dt.fit(x, y)

obj = [4, 3, 1, 0]
p = dt.predict([obj])
print('this is ', p)

dt_data = tree.export_graphviz(
    dt,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True)

graph = graphviz.Source(dt_data)
graph.render('iris_tree')

def main():
    pass
