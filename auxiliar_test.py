
# =============================================================================================================================== #

import numpy as np
import cv2
import os

from class_vic import self_

# =============================================================================================================================== #

font = cv2.FONT_HERSHEY_SIMPLEX

def retocar(self, img):
    
    zeros = np.zeros([self.dim_fil,self.dim_col,3])
    im_sha_1, im_sha_2, _ = img.shape
    
    if im_sha_1 >= self.dim_fil:
        if im_sha_2 >= self.dim_col:
            zeros = cv2.resize(img,(self.dim_col,self.dim_fil))
        else:
            zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,self.dim_fil))
    elif im_sha_2 >= self.dim_col:
        zeros[0:im_sha_1,:,:] = cv2.resize(img,(self.dim_col,im_sha_1))
    else:
        zeros[0:im_sha_1, 0:im_sha_2,:] = img

    return zeros

def normalizar_imagen(img):

    return img/255 * 2 - 1


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.clase = int()
        self.prob = float()

def overlap_c(x1, w1 , x2 , w2):
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1,l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left

def box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh):
    w = overlap_c(ax, aw, bx, bw)
    h = overlap_c(ay, ah, by, bh)
    if w < 0 or h < 0: return 0
    area = w * h
    return area

def box_union_c(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw * ah + bw * bh -i
    return u

def box_iou_c(ax, ay, aw, ah, bx, by, bw, bh):
    return box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh) / box_union_c(ax, ay, aw, ah, bx, by, bw, bh)

def expit_c(x):
    return 1/(1+np.exp(-np.clip(x,-10,10)))

def NMS(self, final_probs , final_bbox):

    labels = self.labels
    C = self.C
    
    boxes = []
    indices = []
  
    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]

    for class_loop in range(class_length):
        for index in range(pred_length):

            if final_probs[index,class_loop] == 0: continue

            for index2 in range(index+1,pred_length):
                if final_probs[index2,class_loop] == 0: continue
                if index==index2: continue

                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.1:
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] = 0
                        break
                    final_probs[index2,class_loop]=0
            if index not in indices:

                bb=BoundBox(C)

                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]

                bb.prob = final_probs[index,class_loop]
                bb.clase = class_loop

                boxes.append(bb)
                indices.append(index)
                
    return boxes

def new_NMS(self, obj):

    if obj == []:
        return []
    
    sizeObjetos = len(obj)
    indices = []
    for index in range(sizeObjetos):
        for index2 in range(index+1, sizeObjetos):

            if (index2 in indices) or (index in indices): continue 

            if box_iou_c(obj[index].x,obj[index].y,obj[index].w,obj[index].h,obj[index2].x,obj[index2].y,obj[index2].w,obj[index2].h) > 0.1 and obj[index].clase == obj[index2].clase:
                if obj[index].prob > obj[index2].prob:
                    indices.append(index2)
                else:
                    indices.append(index)

    newObjetos = []
    for index in range(sizeObjetos):
        if index in indices: continue
        newObjetos.append(obj[index])

    
    return newObjetos
    

def box_constructor(self, net_out_in):

    threshold = self.threshold
    anchors = self.anchors

    H, W = self.H, self.W
    B, C = self.B, self.C

    Classes = net_out_in[:,:,:,:C].reshape([H, W, C])
    Confs_pred = net_out_in[:,:,:,C:2*C].reshape([H, W, C])
    Bbox_pred = net_out_in[:,:,:,2*C:].reshape([H, W, C, 4])
    
    probs = np.zeros((H, W, C), dtype=np.float32)
    _Bbox_pred = np.zeros((H, W, C, 5), dtype=np.float32)
    
    boxes = []
    for row in range(H):
        for col in range(W):

            Classes[row, col, :] = expit_c(Classes[row, col, :])
            if np.max(Classes[row, col, :]) < threshold:
                continue

            Confs_pred[row, col, :] = expit_c(Confs_pred[row, col, :])
            if np.max(Confs_pred[row, col, :]) < threshold:
                continue

            for class_loop in range(C):
                
                tempc = Classes[row, col, class_loop] * Confs_pred[row, col, class_loop]
                if(tempc > threshold):
                    
                    bb=BoundBox(C)

                    bb.x = (col + expit_c(Bbox_pred[row, col, class_loop, 0])) / W
                    bb.y = (row + expit_c(Bbox_pred[row, col, class_loop, 1])) / H
                    bb.w = np.exp(np.clip(Bbox_pred[row, col, class_loop, 2],-15,8)) / W
                    bb.h = np.exp(np.clip(Bbox_pred[row, col, class_loop, 3],-15,8)) / H

                    bb.prob = tempc
                    bb.clase = class_loop

                    boxes.append(bb)

    return new_NMS(self, boxes)

def box_constructor_sin_nms(self, net_out_in):

    threshold = self.threshold
    labels = self.labels
    anchors = self.anchors

    H, W = self.H, self.W
    C = self.C
    
    boxes = []

    Classes = net_out_in[:,:,:,:C].reshape([H, W, C])
    Confs_pred = net_out_in[:,:,:,C:2*C].reshape([H, W, C])
    Bbox_pred = net_out_in[:,:,:,2*C:].reshape([H, W, C, 4])
    
    for row in range(H):
        for col in range(W):
            
            Classes[row, col, :] = expit_c(Classes[row, col, :])
            if np.max(Classes[row, col, :]) < threshold:
                continue

            Confs_pred[row, col, :] = expit_c(Confs_pred[row, col, :])
            if np.max(Confs_pred[row, col, :]) < threshold:
                continue

            for class_loop in range(C):
                
                tempc = Classes[row, col, class_loop] * Confs_pred[row, col, class_loop]
                if(tempc > threshold):

                    bb=BoundBox(C)

                    bb.x = (col + expit_c(Bbox_pred[row, col, class_loop, 0])) / W
                    bb.y = (row + expit_c(Bbox_pred[row, col, class_loop, 1])) / H
                    bb.w = np.exp(np.clip(Bbox_pred[row, col, class_loop, 2],-15,8)) / W
                    bb.h = np.exp(np.clip(Bbox_pred[row, col, class_loop, 3],-15,8)) / H

                    bb.prob = tempc
                    bb.clase = class_loop

                    boxes.append(bb)

    return boxes

def findboxes(self, net_out):
    
    boxes = []
    if self.nms:
        boxes = box_constructor(self, net_out)
    else:
        boxes = box_constructor_sin_nms(self, net_out)
    
    return boxes

def process_box(self, b, h, w):

    left  = int ((b.x - b.w/2.) * w)
    right = int ((b.x + b.w/2.) * w)
    top   = int ((b.y - b.h/2.) * h)
    bot   = int ((b.y + b.h/2.) * h)
    if left  < 0    :  left = 0
    if right > w - 1: right = w - 1
    if top   < 0    :   top = 0
    if bot   > h - 1:   bot = h - 1

    return (left, right, top, bot, self.labels[b.clase], b.prob)


def postprocess(self, net_out, im, h, w):

    labels = self.labels
    colors = self.colors

    boxes = findboxes(self, net_out)
    
    imgcv = im.astype('uint8')
    resultsForJSON = []
    for b in boxes:
        
        boxResults = process_box(self, b, h, w)
        if boxResults is None:
            continue
        
        left, right, top, bot, mess, confidence = boxResults
        resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})  
        
        cv2.rectangle(imgcv,(left, top), (right, bot),colors[labels.index(mess)], 2)
            
        confi = confidence*100

        if self.ver_probs:
            if top - 16 > 0:
                cv2.rectangle(imgcv,(left-1, top - 16), (left + (len(mess)+9)*5*2-1, top),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
            else:
                cv2.rectangle(imgcv,(left-1, top), (left + (len(mess)+9)*5*2-1, top+16),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
        else:
            if top - 16 > 0:
                cv2.rectangle(imgcv,(left-1, top - 16), (left + len(mess)*5*2-1, top),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
            else:
                cv2.rectangle(imgcv,(left-1, top), (left + len(mess)*5*2-1, top+16),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

    return resultsForJSON, imgcv


""" ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """
""" ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """
""" ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """
