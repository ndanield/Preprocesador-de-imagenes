import operator
import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread("test-image.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

#cv2.drawContours(mask,contours,0,255,-1)
#pixelpoints = np.transpose(np.nonzero(mask))

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#encuentra el contorno de mayor area
areas_cnt = [cv2.contourArea(x) for x in contours]
max_idx, max_area = max(enumerate(areas_cnt), key=operator.itemgetter(1))
cnt=contours[max_idx]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
mask = np.zeros((h,w), np.uint8)
mask = thresh[y:y+h, x:x+w]
print mask
onpix = cv2.countNonZero(mask)

#el rectangulo se dibuja en im pero se calcula el contorno en thresh
cv2.imwrite("b.png", mask)

print "letter a"
print "x-box", x
print "y-box", y
print "width", w
print "high", h
print "onpix", onpix
print
