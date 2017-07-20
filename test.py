import operator
import numpy as np
import cv2
from matplotlib import pyplot as plt

def x_bar(matrix):
    sm = 0.0
    width = matrix.shape[1]
    centro = width/2 - 1
    for fila in matrix:
        for i, col in enumerate(fila):
            if col > 0:
                sm += i - centro
    return sm/width

def y_bar(matrix):
    sm = 0.0
    alto = matrix.shape[0]
    centro = alto/2 - 1
    for j, fila in enumerate(matrix):
        for col in fila:
            if col > 0:
                sm += j - centro
    return sm/alto

im = cv2.imread("test-image.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

#cv2.drawContours(mask,contours,0,255,-1)
#pixelpoints = np.transpose(np.nonzero(mask))

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#encuentra el contorno de mayor area
areas_cnt = [cv2.contourArea(x) for x in contours]
print "areas", areas_cnt

#itemgetter(1) quiere decir que toma en cuenta el segundo elento de la tupla que devuelve el enumerate
print "enumerando..."
enumeradas = enumerate(areas_cnt)
max_idx, max_area = max(enumeradas, key=operator.itemgetter(1))
print "mayor index", max_idx
cnt=contours[max_idx]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
mask = np.zeros((h,w), np.uint8)
mask = thresh[y:y+h, x:x+w]

print mask
onpix = cv2.countNonZero(mask)
xbar = x_bar(mask)
ybar = y_bar(mask)

#el rectangulo se dibuja en im pero se calcula el contorno en thresh
cv2.drawContours(im,contours,-1,255,-1)
cv2.imwrite("b.png", im)

print "letter a"
print "x-box", x
print "y-box", y
print "width", w
print "high", h
print "onpix", onpix
print "x-bar", xbar
print "y-bar", ybar
