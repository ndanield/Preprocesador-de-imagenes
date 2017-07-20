import operator
import numpy as np
import cv2
from matplotlib import pyplot as plt

#para que quede claro, 'bar' se refiere a que la variable es una media (lleva la rallita arriba)
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

def x2_bar(matrix):
    sm = 0.0
    width = matrix.shape[1]
    centro = width/2 - 1
    for fila in matrix:
        for i, col in enumerate(fila):
            if col > 0:
                sm += (i - centro) ** 2
    return sm/width

def y2_bar(matrix):
    sm = 0.0
    alto = matrix.shape[0]
    centro = alto/2 - 1
    for j, fila in enumerate(matrix):
        for col in fila:
            if col > 0:
                sm += (j - centro) ** 2
    return sm/alto

def x2_ybr(matrix):
    sm = 0.0
    width = matrix.shape[1]
    centro = width/2 - 1
    for fila in matrix:
        for i, col in enumerate(fila):
            if col > 0:
                varianza = (i - centro) ** 2
                sm += varianza * (i - centro)
    return sm/width

def x_y2br(matrix):
    sm = 0.0
    alto = matrix.shape[0]
    centro = alto/2 - 1
    for j, fila in enumerate(matrix):
        for col in fila:
            if col > 0:
                varianza = (j - centro) ** 2
                sm += varianza * (j - centro)
    return sm/alto

def xy_bar(matrix):
    sm = 0.0
    alto = matrix.shape[0]
    ancho = matrix.shape[1]
    n = alto * ancho
    centrox = ancho/2 - 1
    centroy = alto/2 - 1
    flipmat = matrix[::-1]
    for j, fila in enumerate(flipmat):
        for i, col in enumerate(fila):
            if col > 0:
                sm += (j - centroy) * (i - centrox)
    return sm/n

def x_ege(matrix):
    suma = 0
    for fila in matrix:
        for i, col in enumerate(fila):
            if col > 0:
                if i == 0:
                    suma += 1
                elif fila[i-1] == 0:
                    suma += 1
    return suma

def x_egvy(matrix):
    suma = 0
    #este es recorriendo las filas de abajo hacia arriba, por eso solo flipee el orden de las filas
    flipmat = matrix[::-1]
    for j, fila in enumerate(flipmat):
        for i, col in enumerate(fila):
            if col > 0:
                if i == 0 or fila[i-1] == 0:
                    suma += j
    return suma
    
def y_ege(matrix):
    suma = 0
    flipmat = matrix[::-1]
    for j, fila in enumerate(flipmat):
        for i, col in enumerate(fila):
            if col > 0:
                if j == 0:
                    suma += 1
                elif flipmat[j-1][i] == 0:
                    suma += 1
    return suma

def y_egvy(matrix):
    suma = 0
    flipmat = matrix[::-1]
    for j, fila in enumerate(flipmat):
        for i, col in enumerate(fila):
            if col > 0:
                if j == 0 or flipmat[j-1][i] == 0:
                    suma += i
    return suma

im = cv2.imread("test-image.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

#cv2.drawContours(mask,contours,0,255,-1)
#pixelpoints = np.transpose(np.nonzero(mask))

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#encuentra el contorno de mayor area
areas_cnt = [cv2.contourArea(x) for x in contours]
#print "areas", areas_cnt

#itemgetter(1) quiere decir que toma en cuenta el segundo elento de la tupla que devuelve el enumerate
##print "enumerando..."
enumeradas = enumerate(areas_cnt)
max_idx, max_area = max(enumeradas, key=operator.itemgetter(1))
#print "mayor index", max_idx
cnt=contours[max_idx]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
mask = np.zeros((h,w), np.uint8)
mask = thresh[y:y+h, x:x+w]

print mask
onpix = cv2.countNonZero(mask)
xbar = x_bar(mask)
ybar = y_bar(mask)
x2bar = x2_bar(mask)
y2bar = y2_bar(mask)
x2ybr = x2_ybr(mask)
xy2br = x_y2br(mask)
xybar = xy_bar(mask)
xege = x_ege(mask)
xegvy = x_egvy(mask)
yege = y_ege(mask)
yegvy = y_egvy(mask)


#el rectangulo se dibuja en im pero se calcula el contorno en thresh
cv2.drawContours(im,contours,-1,255,-1)
cv2.imwrite("b.png", im)

print "letter SOMETHING"
print "x-box", x
print "y-box", y
print "width", w
print "high", h
print "onpix", onpix
print "x-bar", xbar
print "y-bar", ybar
print "x2bar", x2bar
print "y2bar", y2bar
print "xybar", xybar
print "x2ybr", x2ybr
print "xy2br", xy2br
print "x-ege", xege
print "xegvy", xegvy
print "y-ege", yege
print "yegvy", yegvy
