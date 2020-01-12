
import os
import numpy as np
import cv2


listaTxt = []
listaJpg = []

for ruta, _, archivos in os.walk("endtoend/"):
    for nombreArchivo in archivos:
        rutaCompleta = os.path.join(ruta, nombreArchivo)

        if(rutaCompleta.endswith("txt")):
            listaTxt.append(rutaCompleta)
        elif(rutaCompleta.endswith("jpg") or rutaCompleta.endswith("png")):
            listaJpg.append(rutaCompleta)


contador = 1
for nameTxt in listaTxt:
    
    if(not os.path.exists(nameTxt.split(".")[0] + ".jpg")):
        continue

    nameJpg = nameTxt.split(".")[0] + ".jpg"
    img = cv2.imread(nameJpg)

    newNameJpg = "car3_img/car3_" + str(contador).zfill(10) + ".jpg"
    
    lineas = []
    with open(nameTxt, 'r') as f:
        for line in f:

            linea = line.split("\t")

            top = int(linea[1])
            bot = int(linea[3]) + top
            left = int(linea[2])
            right = int(linea[4]) + left

            matricula = linea[5].rstrip("\n")

            lineas.append([newNameJpg, top, left, bot, right])

            cv2.imwrite("car3_OCR/" + matricula + ".jpg", img[left:right, top:bot, :])

    with open("car3_label/car3_" + str(contador).zfill(10) + ".txt", "w") as f:
        for line in lineas:
            f.write(line[0] + ',' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + ',' + str(line[4]) + '\n')

    cv2.imwrite(newNameJpg, img)

    contador += 1

    """

    img = cv2.imread(nameJpg)
    cv2.rectangle(img, (top, left), (bot, right), (0,255,0), 2)
    cv2.imshow('jpg', img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    """



