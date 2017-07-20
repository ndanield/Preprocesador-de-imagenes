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


def image_process(im, letter, case):
    results = []
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV) #binary inv! fijate que use esto eh
   
    #concectar espacios separados como por ejemplo en las i minusculas
    se = np.ones((14,14), dtype='uint8')
    #image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
    image_close = cv2.dilate(thresh, se, iterations = 1)
    
    #encuentra los contornos
    _, contours, _ = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
        
        #dibuja el rectangulo en la imagen
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
        
        if case == LOWERCASE:
            cv2.imwrite("boxes/Lower_%s.png" % letter, im)
        elif case == UPPERCASE:
            cv2.imwrite("boxes/Upper_%s.png" % letter, im)
            
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

    return results

if __name__ == "__main__":
    #nochar = int(raw_input("cuantas letras? "))
    abc_size = 26
    completion = 0
    LOWERCASE = 1
    UPPERCASE = 0
    with open("letras.csv", "w") as f:    
        f.write("x-box,y-box,ancho,alto,onpix,x-bar,y-bar,x2bar,y2bar,xybar,x2ybr,xy2br,x-ege,xegvy,y-ege,yegvx,letra\n")        
        for i in range(abc_size):
            filename = chr(ord('A') + i)
            im = cv2.imread("samples/capital/" + filename + ".png")
            results = image_process(im, filename, UPPERCASE)    
            for x in results:
                f.write(",".join(str(s) for s in x) + "\n")
            completion += 1
            print "(%d/%d)" % (completion, abc_size*2)
        for i in range(abc_size):
            filename = chr(ord('a') + i) 
            im = cv2.imread("samples/lower/" + filename + ".png")
            results = image_process(im, filename, LOWERCASE)    
            for x in results:
                f.write(",".join(str(s) for s in x) + "\n")
            completion += 1
            print "(%d/%d)" % (completion, abc_size*2)
    
