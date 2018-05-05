import numpy as np
from sklearn.cluster import AgglomerativeClustering as AggloClust
from sklearn.cluster import mean_shift, estimate_bandwidth
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import mlcourse.misc as misc
from itertools import permutations
import cv2
import os

# classes
colors = list(permutations((0, 128, 255), 3))
targets = [0, 1, 2, 2, 1, 0, 0, 2, 2, 1, 0, 1, 2, 1, 0, 2, 0, 2, 0, 1, 1, 2]

# load an image
dir_path = os.path.dirname(os.path.realpath(__file__))
text_opts = (cv2.FONT_HERSHEY_SIMPLEX, 0.32, (128, 128, 64), 1)
img = cv2.imread('%s/img_black.jpg' % dir_path)

# extract blobs:
blobs = misc.extract_objects(img)

# here we will iterate through every object
# and extract average colour as a feature:

# empty list of features:
features = []

# iterate:
for blob in blobs:
    # extract bounding box
    [x, y, w, h] = cv2.boundingRect(blob)

    # cut out piece of image
    cut = img[y:y + h, x:x + w]
    # extract feature vector
    density = misc.calc_density(cut, blob)
    moments = misc.calc_hu_moments(cut, blob)
    circularity = misc.calc_circularity(cut, blob)
    features.append([
        density,
        circularity,
        moments[0]
    ])

# convert list to numpy array
features = np.asarray(features)
# scale the features to comparable ranges
features = MinMaxScaler().fit_transform(features)

# now use the feature vector for clustering:
# predict = AggloClust(n_clusters=3).fit_predict(features.reshape(-1, 1))
# predict = KMeans(n_clusters=3).fit_predict(features.reshape(-1, 1))
predict = mean_shift(features, estimate_bandwidth(features))[1]

# now draw the classified objects:
# iterate through blobs
for i, blob in enumerate(blobs):
    [x, y, w, h] = cv2.boundingRect(blob)
    cv2.rectangle(img, (x, y), (x + w, y + h), colors[predict[i]], 2)
    txt = '{} d={:.2f},m={:.2f},c={:.2f}'.format(i, *features[i])
    cv2.putText(img, txt, (x + w - 75, y + h + 15), *text_opts)


def pca():
    names = ['circle', 'rectangle', 'triangle']
    f_r = PCA(n_components=2).fit_transform(features)
    y = np.array(targets)
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.scatter(f_r[y == i, 0], f_r[y == i, 1], color=color, alpha=.8,
                    lw=2, label=names[i])
    plt.legend(loc='best', scatterpoints=1)
    plt.title('PCA')
    plt.show()


def main():
    cv2.imshow('blobs', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
