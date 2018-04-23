import numpy as np
from sklearn.cluster import KMeans
import mlcourse.misc as misc
import cv2
import os

# classes
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255),
          (0, 255, 255), (255, 255, 0), (128, 0, 128),
          (128, 128, 0), (0, 128, 128), (128, 128, 128)]

# load an image
dir_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread('%s/img_black.jpg' % dir_path)

# extract blobs:
blobs = misc.extractObjects(img)

# here we will iterate through every object
# and extract average colour as a feature:

# empty list of features:
features = []

# iterate:
for blob in blobs:
    # extract bounding box
    [x, y, w, h] = cv2.boundingRect(blob)

    # cut out piece of image
    cut = img[y:y+h, x:x+w]

    # extract color:
    clr = misc.calcAverageColor(cut)

    # TODO: Think of other features

    features.append(clr)

# convert list to numpy array
features = np.asarray(features)

# now use the feature vector for KMeans:
kmeans = KMeans(n_clusters=3, random_state=235)
labels = kmeans.fit_predict(features.reshape(-1, 1))

# now draw the classified objects:
# iterate through blobs
for i, blob in enumerate(blobs):
    [x, y, w, h] = cv2.boundingRect(blob)
    cv2.rectangle(img, (x, y), (x+w, y+h), colors[labels[i]], 2)


# display the image
cv2.imshow("blobs", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


def main():
    print(__name__)
