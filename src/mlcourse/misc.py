import numpy as np
import cv2


def fillHoles(image, size=50):
    # extract contours
    im2, contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > size:
            cv2.drawContours(image, contours, i, (255, 255, 255), -1)
        else:
            cv2.drawContours(image, contours, i, (0, 0, 0), -1)
    return image


def extractObjects(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholding
    th, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)

    # fill holes
    mask = fillHoles(mask)

    # process blobs:
    # extract contours
    im2, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def calcAverageColor(img):
    # resize
    w, h, d = img.shape

    img = img[int(h / 4):int(h / 4 + h / 2), int(w / 4):int(w / 4 + w / 2)]

    # first convert RGB to HSV and take only hue:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    # histogram
    hist = cv2.calcHist([hue], [0], None, [179], [1, 180])

    # max color:
    maxColor = np.argmax(hist, axis=0)

    return maxColor
