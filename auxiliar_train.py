
# =============================================================================================================================== #

import numpy as np
import cv2
import os

from random import shuffle
from copy import deepcopy

from class_vic import self_

# =============================================================================================================================== #

font = cv2.FONT_HERSHEY_SIMPLEX

def leer_datos_text(ruta):

    image_label_nomb = []

    for ruta, _, ficheros in os.walk(ruta):
        for nombre_fichero in ficheros:
            rut_comp = os.path.join(ruta, nombre_fichero)
            
            if(rut_comp.endswith("txt")):
                image_label_nomb.append(rut_comp)

    return image_label_nomb

def retocar(self, img, cosico):
    
    zeros = np.zeros([self.dim_fil,self.dim_col,3])
    im_sha_1, im_sha_2, _ = img.shape
    
    if im_sha_1 >= self.dim_fil:
        if im_sha_2 >= self.dim_col:
            zeros = cv2.resize(img,(self.dim_col,self.dim_fil))
            for obj in cosico:
                obj[2],obj[1],obj[4],obj[3] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[1]*self.dim_col/im_sha_2), int(obj[4]*self.dim_fil/im_sha_1), int(obj[3]*self.dim_col/im_sha_2)
        else:
            zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,self.dim_fil))
            for obj in cosico:
                obj[2],obj[4] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[4]*self.dim_fil/im_sha_1)
    elif im_sha_2 >= self.dim_col:
        zeros[0:im_sha_1,:,:] = cv2.resize(img,(self.dim_col,im_sha_1))
        for obj in cosico:
            obj[1],obj[3] = int(obj[1]*self.dim_col/im_sha_2), int(obj[3]*self.dim_col/im_sha_2)
    else:
        zeros[0:im_sha_1, 0:im_sha_2,:] = img

    return zeros, cosico


def normalizar_imagen(img):

    return img/255 * 2 - 1

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def _batch(self, img ,allobj):

    H, W = self.H, self.W
    B, C = self.B, self.C
    
    #anchors = self.anchors
    labels = self.labels
    
    cellx = 1. * self.dim_col / W
    celly = 1. * self.dim_fil / H

    y_true_etiqueta = np.zeros([H,W,C*(1+1+4)])
    for obj in allobj:

        if obj[0] not in labels:
            continue
        
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly

        if cx >= W or cy >= H: return []
        
        if obj[1] < 1:
            obj[1] = 1
        if obj[2] < 1:
            obj[2] = 1
        if obj[3] > self.dim_col:
            obj[3] = self.dim_col - 1
        if obj[4] > self.dim_fil:
            obj[4] = self.dim_fil - 1
            
        obj[3] = float(obj[3]-obj[1]) / self.dim_col
        obj[4] = float(obj[4]-obj[2]) / self.dim_fil

        for cog in range(3,5):
            if obj[cog] < 0:
                obj[cog] = 0.001
        
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery

        numb_magic = int(np.floor(cy) * W + np.floor(cx))

        resto = int(numb_magic%W)
        el_otro = int((numb_magic - resto) / W)

        posicionReticulaClase = labels.index(obj[0])
        
        y_true_etiqueta[el_otro, resto, posicionReticulaClase] = 1.
        y_true_etiqueta[el_otro, resto, C + posicionReticulaClase] = 1.
        y_true_etiqueta[el_otro, resto, 2*C + 4*posicionReticulaClase: 2*C + 4*posicionReticulaClase+4] = obj[1:5]
    
    return normalizar_imagen(img), y_true_etiqueta

def agrandar_bboxes(self_, image, bboxes):

    H, W = self_.H, self_.W

    sha1, sha2, _ = image.shape

    cellx = 1. * sha2 / W
    celly = 1. * sha1 / H

    new_bboxes = deepcopy(bboxes)
    for box in bboxes:

        mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

        centerx = .5*(left+right)
        centery = .5*(top+bot)
        cx = centerx / cellx
        cy = centery / celly
        
        ml_x, ml_y = (right - left)/2, (bot - top)/2

        for i, j in [(0,1), (1,1), (1,0), (0,-1), (-1,-1), (-1,0), (1,-1), (-1,1)]:
            
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

                new_left, new_right = int(new_cx*cellx - ml_x), int(new_cx*cellx + ml_x)
                new_top, new_bot = int(new_cy*celly - ml_y), int(new_cy*celly + ml_y)

                if iou([new_left, new_top, new_right, new_bot],[left, top, right, bot]) < 0.5:
                    continue

                new_bboxes.append([mess, new_left, new_top, new_right, new_bot])

    return new_bboxes
    
def leer_imagen_en_eg_o_color(nombre):

    if True:#np.random.randint(10)%2:

        return cv2.imread(nombre)

    else:
        
        eg = cv2.imread(nombre, 0)
        largo, ancho = eg.shape

        eg_triple = np.zeros([largo, ancho,3])

        eg_triple[:,:,0] = eg
        eg_triple[:,:,1] = eg
        eg_triple[:,:,2] = eg

        return eg_triple.astype('uint8')

def retocar_imagen_y_coordenadas(imagen, bboxes):

    if False:#np.random.randint(10)%2:
        sha_y,sha_x,_= imagen.shape
        noise = np.random.rand(sha_y,sha_x,3)
        imagen = imagen + noise*np.random.randint(3,10)

    return imagen, bboxes

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

        #final_bbox = agrandar_bboxes(self, transformed_image,final_bbox)

        im_de_out_batch, y_true_ = _batch(self, transformed_image ,final_bbox)
        
        im_train.append(im_de_out_batch)
        y_true_nnp.append(y_true_)

    return im_train, y_true_nnp