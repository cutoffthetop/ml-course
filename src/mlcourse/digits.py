import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# load digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

# encode to using one-hot label encoder
encoder = LabelEncoder()
encoder.fit(y)
y = to_categorical(encoder.transform(y))

# in-place scaling of features to range(0, 1)
MinMaxScaler(copy=False).fit_transform(x)

# shuffle and split into training and test
x, x_test, y, y_test = train_test_split(
    x, y,
    test_size=0.25,
    shuffle=True,
    random_state=42)

# fit multi-layer perceptron classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 32, 10),
    activation='relu',
    solver='adam',
    max_iter=250,
    random_state=42,
    verbose=True)
mlp = mlp.fit(x, y)

# output mean mlp accuracy on test data
mlp_accuracy = mlp.score(x_test, y_test)
print('sklearn', mlp_accuracy)

# construct multi-layer keras network
seq = Sequential()
seq.add(Dense(32, activation='relu', input_dim=64))
seq.add(Dense(32, activation='relu'))
seq.add(Dense(10, activation='softplus'))
seq.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])
seq.fit(x, y, epochs=250)

_, seq_accuracy = seq.evaluate(x_test, y_test)
print('keras', seq_accuracy)

def main():
    print(__name__)
