from random import shuffle

from auxiliar_test import postprocess, retocar
from neuralnetwork import mark1, loss_function
from class_vic import self_ as self

import time
import numpy as np
import cv2
import os

import tensorflow as tf
import sys

def eliminar_elementos(ruta):
    lista_elementos = []
    for ruta, _, ficheros in os.walk(ruta):
        for nombre_fichero in ficheros:
            rut_comp = os.path.join(ruta, nombre_fichero)
            lista_elementos.append(rut_comp)
            
    for eliminando in lista_elementos:
        os.remove(eliminando)

if os.path.exists(self.h5):
    print('')
    print('Cargando modelo')
    print('')
    model = tf.keras.models.load_model(self.h5, custom_objects={'loss_function': loss_function})
else:
    model, h_out = mark1(self)
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))

print('')
print(model.summary())
print('')

salir = 'nada_de_eso'
while salir not in ['out', 'salir', 'exit', 'irse', 'ir']:
    
    print('Hola, buenas.')
    nhoa = input('Que quieres analizar, video, foto, usar la webcam o grabar video (v/f/c/g): ')

    lista_posibilidades = ['v', 'video', 'Video','imagen','Imagen', 'f', 'foto', 'Foto', 'fotografia', 'c', 'wc', 'webcam', 'g', 'grabar', 'r', 'record']
    cabrear = 0

    while nhoa not in lista_posibilidades:

        print('')
        nhoa = input('Que quieres hacer, video, foto, webcam o grabar con la webcam: ')

        if cabrear == 5:

            print('')
            print('Me estás cabreando introduce bien la maldita respuesta, video, foto, webcam o grabar con la webcam')

        if cabrear > 10:

            print('')
            print('Hala ya me has cabreado. Adios.')
            print('')
            sys.exit()

        cabrear += 1

    print('')

    font = cv2.FONT_HERSHEY_SIMPLEX
    if nhoa in ['v', 'video', 'Video']:

        image_label_nomb = []
        for ruta, subdirs, ficheros in os.walk('/home/sergio/Vídeos/'):
            for nombre_fichero in ficheros:
                rut_comp = os.path.join(ruta, nombre_fichero)
                if (rut_comp.endswith(".avi") or rut_comp.endswith(".mp3") or rut_comp.endswith(".mp4") or rut_comp.endswith(".mpg") or rut_comp.endswith(".AVI")) \
                   and ('anali' not in rut_comp) and ('graba_vid' not in rut_comp):
                    image_label_nomb.append(rut_comp)

        if image_label_nomb != []:
            print('')
            print('He encontrado unos cuantos videos por aqui. ')
            respues_videos = input('Quieres verlos y escoges cual analizo (s: quiero ver / no: no quiero ver):')
            print('')

            if respues_videos in ['s', 'S', 'Y', 'y', 'si', 'Si', 'SI', 'sI', 'Yes', 'yes', 'YES']:
                count = 1
                for a in image_label_nomb:
                    print('     ', str(count), ' -> ', a)
                    count+=1
                print('')
                respues_videos_2 = input('Elige uno colega: ')
                try:
                    int(respues_videos_2)
                except:
                    print('')
                    print('Que haces. Has introducido una mala respuesta.')
                    print('Me has cabreado. Adios.')
                    print('')
                    sys.exit()
                if int(respues_videos_2) > count or int(respues_videos_2) < 1:

                    print('')
                    print('Que haces. Has introducido una mala respuesta.')
                    print('Me has cabreado. Adios.')
                    print('')
                    sys.exit()

                else:
                    ruta_dada = image_label_nomb[int(respues_videos_2)-1]
                    print('')

            else:

                print('')
                print('===================================================================================================================')
                ruta_dada = input('     Introduzca la dirección donde se almacenan el video (no olvides el .avi, .mp3 o lo que sea): ')
                print('===================================================================================================================')
                print('')

        else:

            print('')
            print('===================================================================================================================')
            ruta_dada = input('     Introduzca la dirección donde se almacenan el video (no olvides el .avi, .mp3 o lo que sea): ')
            print('===================================================================================================================')
            print('')

        if not os.path.exists(ruta_dada):

            print('')
            print('************ATENCION --> RUTA DADA: "',ruta_dada,'" NO EXISTENTE************')
            print('')
            print('                  ADIOS ADIOS ADIOS ADIOS ADIOS ADIOS')
            print('')
            sys.exit()

        if not os.path.exists("analizados_videos"):
            os.mkdir("analizados_videos")
            
        nombre_video_guarda = 'analizados_videos/' + os.path.basename(ruta_dada)
        
        cap = cv2.VideoCapture(ruta_dada)

        step = 0
        cont = 0
        mi_primera_vez = True
        ancho = False
        while True:
            if (cap.isOpened()== False):
                print('')
                print("Error opening video stream or file")
                print('')
                break
                #sys.exit()
            else:
                try:
                    start_time = time.time() # start time of the loop
        
                    ret, frame = cap.read()
                    #frame = rotateImage(frame, 90)
                    if mi_primera_vez:
                        orig_y, orig_x, _ = frame.shape
                        if orig_x > 1500:
                            orig_x = 1024
                            orig_y = 720
                            ancho = True
                            
                        mult_y, mult_x = orig_y, orig_x
                        if orig_y < self.dim_fil:
                            mult_y = self.dim_fil
                        if orig_x < self.dim_col:
                            mult_x = self.dim_col
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out_video = cv2.VideoWriter(nombre_video_guarda,fourcc, 20.0, (orig_x,orig_y))
                        mi_primera_vez = False

                    if ancho:
                        frame = cv2.resize(frame, (orig_x, orig_y))
                        
                    frame_adaptado = retocar(self, frame)
                    frame_norm = (frame_adaptado/255)*2 - 1
                    
                    salida = model.predict(x=np.array([frame_norm]))
                    cajitas, img_out = postprocess(self,salida, frame, mult_y, mult_x)

                    #new_image = np.zeros([orig_x,2*orig_y+5,3])
                    #new_image[:,0:orig_y,:] = frame
                    #new_image[:,orig_y+5:,:] = img_out
                    if step%20 == 0:
                        print('Tiempo de video analizado (en segundos): ',cont)
                        cont += 1
                    step+=1
                    
                    out_video.write(img_out.astype('uint8'))
                    cv2.imshow('frame',img_out/255)

                    el_tiempo = np.zeros([32,180,3])
                    cv2.putText(el_tiempo,"FPS: " + str(1.0 / (time.time() - start_time)) ,(5, 15), font, 0.55,(255,255,255),0,cv2.LINE_AA)
                    cv2.imshow('FPS',el_tiempo/255)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        
                        #cap.release()
                        #out_video.release()
                        
                        #sys.exit()
                        break

                except:
                    print('')
                    print('     -----> TERMINADO <-----')
                    print('')
                    cap.release()
                    out_video.release()
                    #sys.exit()

        print('')
        print('     -----> TERMINADO <-----')
        print('')
        cv2.destroyAllWindows()
        cap.release()
        out_video.release()

    elif nhoa in ['c', 'wc', 'webcam']:

        cap = cv2.VideoCapture(-1)

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
                if orig_y < self.dim_fil:
                    mult_y = self.dim_fil
                if orig_x < self.dim_col:
                    mult_x = self.dim_col
                mi_primera_vez = False
                
            frame_adaptado = retocar(self, frame)
            frame_norm = (frame_adaptado/255)*2 - 1

            salida = model.predict(x=np.array([frame_norm]))
            cajitas, img_out = postprocess(self,salida, frame, mult_y, mult_x)
                    
            #new_image = np.zeros([orig_x,2*orig_y+5,3])
            #new_image[:,0:orig_y,:] = frame
            #new_image[:,orig_y+5:,:] = img_out
            
            cv2.imshow('Entrada/Salida',img_out/255)

            el_tiempo = np.zeros([32,180,3])
            cv2.putText(el_tiempo,"FPS: " + str(1.0 / (time.time() - start_time)) ,(5, 15), font, 0.55,(255,255,255),0,cv2.LINE_AA)
            cv2.imshow('FPS',el_tiempo/255)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    elif nhoa in ['g', 'grabar', 'r', 'record']:

        if not os.path.exists("graba_vid"):
            os.mkdir("graba_vid")

        print('')
        print('===================================================================================================================')
        nombre = input('     Introduzca el nombre del video: ')
        print('===================================================================================================================')
        print('')

        cap = cv2.VideoCapture(-1)

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
                if orig_y < self.dim_fil:
                    mult_y = self.dim_fil
                if orig_x < self.dim_col:
                    mult_x = self.dim_col
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_video = cv2.VideoWriter('graba_vid/' + nombre + '.avi',fourcc, 13.0, (orig_x,orig_y))
                mi_primera_vez = False
                
            frame_adaptado = retocar(self, frame)
            frame_norm = (frame_adaptado/255)*2 - 1

            salida = model.predict(x=np.array([frame_norm]))
            cajitas, img_out = postprocess(self,salida, frame, mult_y, mult_x)
                    
            #new_image = np.zeros([orig_x,2*orig_y+5,3])
            #new_image[:,0:orig_y,:] = frame
            #new_image[:,orig_y+5:,:] = img_out

            out_video.write(img_out.astype('uint8'))
            cv2.imshow('Entrada/Salida',img_out/255)

            el_tiempo = np.zeros([32,180,3])
            cv2.putText(el_tiempo,"FPS: " + str(1.0 / (time.time() - start_time)) ,(5, 15), font, 0.55,(255,255,255),0,cv2.LINE_AA)
            cv2.imshow('FPS',el_tiempo/255)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        
    else:

        ruta_de_ahora_donde_estamos = os.getcwd()
        lista_ruta_de_ahora_donde_estamos = ''
        for ele_de_winni_de_poo in ruta_de_ahora_donde_estamos:
            if ele_de_winni_de_poo == '\\':lista_ruta_de_ahora_donde_estamos += '/'
            else:lista_ruta_de_ahora_donde_estamos+=ele_de_winni_de_poo
        print('')
        print(' Ruta de la carpeta actual -> ', lista_ruta_de_ahora_donde_estamos)
        print('          --> Ruta guay: /home/sergio/Escritorio/Deep learning/0_Base_de_datos/COCO/train2017/')
        print('          --> Ruta guay: /home/sergio/Escritorio/Deep learning/0_Base_de_datos/COCO/val2017/')
        print('')
        print('============================================================================')
        ruta_dada = input('     Introduzca la dirección donde se almacenan las imágenes: ')
        print('============================================================================')
        print('')

        if os.path.exists(ruta_dada):

            quiero_guardar = True
            quiero_mostrar = True
            if quiero_guardar:
                if not os.path.exists("analiza_img"):
                    os.mkdir("analiza_img")
                else:
                    eliminar_elementos('analiza_img/')
                
            image_label_nomb = []

            for ruta, subdirs, ficheros in os.walk(ruta_dada):
                for nombre_fichero in ficheros:
                    rut_comp = os.path.join(ruta, nombre_fichero)
                    if rut_comp.endswith(".jpg") or rut_comp.endswith(".JPG") or rut_comp.endswith(".png") or rut_comp.endswith(".jepg"):
                        image_label_nomb.append(rut_comp)

            image_label_nomb = image_label_nomb[:1000]

            que_sera = input('Quieres desordenar las imagenes: ')
            if que_sera in ['s', 'y', 'yes', 'Y', 'S', 'Si', 'SI', 'si', 'Yes', 'YES'] :
                shuffle(image_label_nomb)
            longi = len(image_label_nomb)
            for i in range(longi):

                frame = cv2.imread(image_label_nomb[i])
                
                orig_y, orig_x, _ = frame.shape
                mult_y, mult_x = orig_y, orig_x
                if orig_y < self.dim_fil:
                    mult_y = self.dim_fil
                if orig_x < self.dim_col:
                    mult_x = self.dim_col
                    
                frame_adaptado = retocar(self,cv2.imread(image_label_nomb[i]))
                frame_norm = (frame_adaptado/255)*2 - 1

                salida = model.predict(x=np.array([frame_norm]))
                cajitas, img_out = postprocess(self,salida, frame, mult_y, mult_x)

                #new_image = np.zeros([orig_x+30,orig_y+30,3])
                #new_image[15:-15,15:orig_y+15,:] = img_out

                if quiero_guardar:
                    cv2.imwrite('analiza_img/imagen-' + str(i) + '.jpg', img_out)
                if quiero_mostrar:
                    cv2.imshow('imagen_resultante', img_out/255)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                        #sys.exit()
                        
                    #cv2.waitKey(0)
                    cv2.destroyAllWindows()

        else:

            print('')
            print('************ATENCION --> RUTA DADA: "',ruta_dada,'" NO EXISTENTE************')
            print('')
            #print('                 ADIOS ADIOS ADIOS ADIOS ADIOS ADIOS')
            #print('')
            #sys.exit()
        

    cv2.destroyAllWindows()

    print('')
    print('Hola, de nuevo.')
    salir = input('Quieres que hagamos algo mas o terminamos ya (cualquuier tecla/exit): ')

            
