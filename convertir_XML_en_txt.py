import xml.etree.ElementTree as ET

import os
import numpy as np
import cv2


listaXml = []

for ruta, _, archivos in os.walk("imageLabel/"):
    for nombreArchivo in archivos:
        rutaCompleta = os.path.join(ruta, nombreArchivo)

        if(rutaCompleta.endswith("xml")):
            listaXml.append(rutaCompleta)

for name in listaXml:        

    tree = ET.parse(name)
    root = tree.getroot()

    ruta = root.find("./folder").text + '/' + root.find("./filename").text

    lineas = []
    for elem in root.findall("./object"):

        left = elem.find("./bndbox/xmin").text
        top = elem.find("./bndbox/ymin").text
        right = elem.find("./bndbox/xmax").text
        bot = elem.find("./bndbox/ymax").text

        lineas.append(ruta + ',' + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bot) + '\n')

    unir = name.split('.')[:-1]
    nombreCompleto = ''
    for kuxi in unir:
        nombreCompleto += kuxi
    nombreCompleto = nombreCompleto + '.txt'

    print(nombreCompleto)
    with open(nombreCompleto, 'w') as f:

        for line in lineas:
            f.write(line)
    """
    img = cv2.imread("/home/sergio/Escritorio/Deep learning/Modelos de visión/Reconocimiento de matrículas/dataset/imageTrain/" + lineas[0].split(',')[0])
    for line in lineas:

        left = int(line.split(',')[1])
        top = int(line.split(',')[2])
        right = int(line.split(',')[3])
        bot = int(line.split(',')[4])

        cv2.rectangle(img, (top, left), (bot, right), (0,255,0), 2)
        cv2.imshow('jpg', img)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    """