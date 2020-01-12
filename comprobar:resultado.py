
import os
import numpy as np
import cv2


listaTxt = []

for ruta, _, archivos in os.walk("car3_label/"):
    for nombreArchivo in archivos:
        rutaCompleta = os.path.join(ruta, nombreArchivo)
        listaTxt.append(rutaCompleta)


for nameTxt in listaTxt:

    lineas = []
    with open(nameTxt, 'r') as f:
        for line in f:
            lineas.append(line.rstrip("\n").split(','))

    img = cv2.imread(lineas[0][0])

    for line in lineas:

        cv2.rectangle(img,(int(line[1]),int(line[2])),(int(line[3]),int(line[4])), (0,255,0), 2)

    cv2.imshow('jpg', img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break