# -*- coding: cp1252 -*-
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


def image_process(im, letter):
    results = []
    
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV) #binary inv! fijate que use esto eh
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #encuentra el contorno de mayor area
    areas_cnt = [cv2.contourArea(x) for x in contours]
    #itemgetter(1) quiere decir que toma en cuenta el segundo elento de la tupla que devuelve el enumerate
    ###enumeradas = enumerate(areas_cnt)
    ###max_idx, max_area = max(enumeradas, key=operator.itemgetter(1))
    ###cnt=contours[max_idx]
    #saca las propiedades del recangulo para ese contorno
    for cnt in contours:
        
        x,y,w,h = cv2.boundingRect(cnt)
        #dibuja el rectangulo en la imagen
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.imwrite("result%s.png" % letter, im)
        mask = np.zeros((h,w), np.uint8) # crea una nueva matris llena de zeros
        mask = thresh[y:y+h, x:x+w] # toma una region de la imagen thresh
        #mask = cv2.resize(mask, (64, 64)) #demosle un tamano normalizado

        onpix = cv2.countNonZero(mask) #a causa de esta funcion es que usamos thresh para invertir colores
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

        results.append((x, y, w, h, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybr, xy2br,xege, xegvy, yege, yegvy, letter))
    '''
    print "letter", letter
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
    '''
    return results, mask


from pprint import pprint

filename = raw_input("filename: ")

im = cv2.imread(filename + ".png")
results, mask = image_process(im, filename)
with open("file.csv", "a") as f:
    #f.write("x-box,y-box,ancho,alto,onpix,x-bar,y-bar,x2bar,y2bar,xybar,x2ybr,xy2br,x-ege,xegvy,y-ege,yegvx,letra\n")
    for x in results:
        f.write(",".join(str(s) for s in x) + "\n")
