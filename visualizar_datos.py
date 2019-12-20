
import numpy as np
import cv2
import os

from random import shuffle
from copy import deepcopy

from class_vic import self_
from auxiliar_train import leer_datos_text, leer_imagen_en_eg_o_color, retocar, retocar_imagen_y_coordenadas, iou

font = cv2.FONT_HERSHEY_SIMPLEX

def dibujar_imagen_y_coordenadas(self_, image, final_bbox):

    for box in final_bbox:
        mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
        
        cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
            
        if top - 16 > 0:
            cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
            cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        else:
            cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
            cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

def visualiza(self_):

    image_label_nomb = leer_datos_text(ruta = self_.rpe)
    shuffle(image_label_nomb)

    for name in image_label_nomb:
        """ Abrimos los .txt y leemos nombre de la imagen tamaño y las cajas. Le pasamos estos datos _batch """
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                if linea[1] in self_.labels:
                    vector.append(linea)

        if vector == []:
            continue

        image = cv2.imread(self_.rpi + vector[0][0])

        bboxes = []
        for mini_vector in vector:
            bboxes.append([float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4]),float(mini_vector[5])])

        final_bbox = []
        coooount = 0
        for box in bboxes:
            if int(box[0]) != 0 or int(box[1]) != 0 or int(box[2]) != 0 or int(box[3]) != 0:
                final_bbox.append([vector[coooount][1], box[0], box[1], box[2], box[3]])
            coooount += 1

        for box in final_bbox:
            mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
            
            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        cv2.imshow('image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False

#Cosa para visualizar
#visualiza(self_)

def programa_para_cargar_lote(self, nom_lot):

    im_train = []
    y_true_nnp = []

    for name in nom_lot:
        
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                if linea[1] in self_.labels:
                    vector.append(linea)

        if vector == []:
            return [], []

        image = leer_imagen_en_eg_o_color(self_.rpi + vector[0][0])
        bboxes = []
        for mini_vector in vector:
            bboxes.append([mini_vector[1], float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4]),float(mini_vector[5])])
            
        image, bboxes = retocar_imagen_y_coordenadas(image, bboxes)
        transformed_image, final_bbox = retocar(self, image,bboxes)

        return transformed_image, bboxes

#Programa para visualizar la funcion cargar_lote
if False:

    image_label_nomb = leer_datos_text(ruta = self_.rpe)

    for name in image_label_nomb:

        image, bboxes = programa_para_cargar_lote(self_, [name])

        shuffle(bboxes)
        for box in bboxes:

            mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

            #print((right - left)*(bot - top))

            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                    
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        cv2.imshow('image', image.astype('uint8'))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

if False:

    H, W = self_.H, self_.W
    B, C = self_.B, self_.C
    
    image_label_nomb = leer_datos_text(ruta = self_.rpe)
    shuffle(image_label_nomb)

    for name in image_label_nomb:
        
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                if linea[1] in self_.labels:
                    vector.append(linea)

        if vector == []:
            print("Ezebez")
            continue

        image = leer_imagen_en_eg_o_color(self_.rpi + vector[0][0])
        bboxes = []
        for mini_vector in vector:
            bboxes.append([mini_vector[1], float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4]),float(mini_vector[5])])
            
        image, bboxes = retocar_imagen_y_coordenadas(image, bboxes)
        image, bboxes = retocar(self_, image,bboxes)
        
        ###########################       VISUALIZACION DE NUEVA FORMA PARA ENTRENAR       ###########################
        sha1, sha2, _ = image.shape

        cellx = 1. * sha2 / W
        celly = 1. * sha1 / H


        tam_1, tam_2 = sha1/H, sha2/W
        for di_fi in range(H):
            for di_co in range(W):
                cv2.rectangle(image,(int(di_co*tam_2), int(di_fi*tam_1)), (int((1+di_co)*tam_2), int((1+di_fi)*tam_1)),(0,0,0), 1)
        
        for box in bboxes:
            mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

            centerx = .5*(left+right) #xmin, xmax
            centery = .5*(top+bot) #ymin, ymax
            cx = centerx / cellx
            cy = centery / celly
            
            ml_x, ml_y = (right - left)/2, (bot - top)/2

            #punto_medio_x, punto_medio_y = int(left + (right - left)/2), int(top + (bot - top)/2)
            #cv2.circle(image,(punto_medio_x,punto_medio_y), 5, (255,0,0), -1)

            for i,j in [(0,1),(1,1),(1,0), (0,-1),(-1,-1),(-1,0), (1,-1), (-1,1)]:
                
                if (cx + i) > 0 and (cx + i) < W and (cy + j) > 0 and (cy + j) < H:

                    new_cx, new_cy = cx + i, cy + j

                    if new_cx < cx:
                        new_cx += np.ceil(cx) - cx
                    elif new_cx > cx:
                        new_cx -= cx  - np.floor(cx)

                    
                    if new_cy < cy:
                        new_cy += np.ceil(cy) - cy
                    elif new_cy > cy:
                        new_cy -= cy  - np.floor(cy)
                    

                    #cv2.circle(image,(int(new_cx*cellx),int(new_cy*celly)), 2, (255,0,255), -1)

                    new_left, new_right = int(new_cx*cellx - ml_x), int(new_cx*cellx + ml_x)
                    new_top, new_bot = int(new_cy*celly - ml_y), int(new_cy*celly + ml_y)

                    if iou([new_left, new_top, new_right, new_bot],[left, top, right, bot]) < 0.6:
                        continue

                    cv2.rectangle(image,(new_left, new_top),(new_right, new_bot),self_.colors[self_.labels.index(mess)], 1)

        ###########################       VISUALIZACION DE NUEVA FORMA PARA ENTRENAR       ###########################

        #for box in bboxes:
            #mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
            
            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            cv2.imshow('image', image.astype('uint8'))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        #break