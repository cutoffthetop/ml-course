from sklearn import datasets
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
x = iris.data
y = iris.target

model = Sequential()
model.add(Dense(5, activation='sigmoid', input_dim=4))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

model.fit(x, dummy_y, epochs=500)


def main():
    print(__name__)
