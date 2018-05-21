import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# load digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

# encode labels to one-hot
y = to_categorical(y)

# scaling of features to fit range 0-1
x = MinMaxScaler().fit_transform(x)

# shuffle and split into training and test
x, x_test, y, y_test = train_test_split(
    x, y,
    test_size=0.25,
    shuffle=True,
    random_state=42)

# fit multi-layer perceptron classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 48, 10),
    activation='relu',
    solver='adam',
    max_iter=250,
    random_state=42,
    verbose=True)
mlp = mlp.fit(x, y)

# output mean mlp accuracy on test data
mlp_accuracy = mlp.score(x_test, y_test)
print('sklearn', mlp_accuracy)

# calculcate confusion matrix for predicted labels
label_pred = np.argmax(mlp.predict(x_test), axis=1)
label_true = np.argmax(y_test, axis=1)
cf_matrix = confusion_matrix(label_true, label_pred)
print(cf_matrix)

# construct multi-layer keras network
seq = Sequential()
seq.add(Dense(48, activation='relu', input_dim=64))
seq.add(Dense(10, activation='softmax'))
seq.compile(
    loss='mean_absolute_error',
    optimizer='adam',
    metrics=['accuracy'])
seq.fit(x, y, epochs=250)

# output mean keras network accuray
_, seq_accuracy = seq.evaluate(x_test, y_test)
print('keras', seq_accuracy)

# calculcate confusion matrix for predicted labels
label_pred = np.argmax(seq.predict(x_test), axis=1)
label_true = np.argmax(y_test, axis=1)
cf_matrix = confusion_matrix(label_true, label_pred)
print(cf_matrix)

def main():
    print(__name__)
