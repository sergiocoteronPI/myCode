
import numpy as np
import cv2
import os

import tensorflow as tf
import sys
from random import shuffle

from auxiliar_1 import postprocess, retocar
from auxiliar_train import leer_datos_text

from ayuda import eliminar_elementos
from mark_1 import mark1, loss_function

from clase_super_importante import self_

import time

self_.colors[0] = [0,255,0]

if os.path.exists(self_.h5):
    model = tf.keras.models.load_model(self_.h5, custom_objects={'loss_function': loss_function})
else:
    model, h_out = mark1(self_)
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))

print('')
print(model.summary())
print('')


font = cv2.FONT_HERSHEY_SIMPLEX
if True:

    cap = cv2.VideoCapture(0)

    step = 0
    cont = 0
    mi_primera_vez = True
    while True:

        if (cap.isOpened()== False):
            print('')
            print("Error opening video stream or file")
            print('')
            break
            #sys.exit()

        start_time = time.time() # start time of the loop
        
        ret, frame = cap.read()
        if mi_primera_vez:
            orig_y, orig_x, _ = frame.shape
            mult_y, mult_x = orig_y, orig_x
            if orig_y < self_.dim_fil:
                mult_y = self_.dim_fil
            if orig_x < self_.dim_col:
                mult_x = self_.dim_col
            mi_primera_vez = False
            
        frame_adaptado, lista_absurda = retocar(self_, frame, [])
        frame_norm = (frame_adaptado/255)*2 - 1

        net_out_ = model.predict(x=np.array([frame_norm]))
        cajitas, img_out = postprocess(self_,net_out_, frame, mult_y, mult_x)

        cv2.imshow('Entrada/Salida',img_out.astype('uint8'))

        el_tiempo = np.zeros([32,180,3])
        cv2.putText(el_tiempo,"FPS: " + str(1.0 / (time.time() - start_time)) ,(5, 15), font, 0.55,(255,255,255),0,cv2.LINE_AA)
        cv2.imshow('FPS',el_tiempo.astype('uint8'))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
                
