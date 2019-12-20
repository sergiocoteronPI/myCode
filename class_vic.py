
import numpy as np

class v_ps:
    def __init__(self, threshold, batch_size, dim_fil, dim_col, H, W, B, learning_ratio, nms, ver_probs, rpe, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                       'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
                       'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                       'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                       'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
                       'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock',
                       'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
        
        #self.anchors = [0.57273,0.677385, 1.87446,2.06253, 3.33843,5.47434, 7.77052,7.16828,  16.62,10.5][0:2*B]
        self.anchors = [1,1, 1,1, 1,1, 1,1, 1,1, 1,1][0:2*B]

        self.H = H
        self.W = W
        self.C = len(self.labels)
        self.B = B
        self.HW = H*W

        self.colors = np.random.randint(0,255 ,(self.C,3)).tolist()
        self.colors[0] = [255,0,255]
        self.learning_ratio = learning_ratio

        self.nms = nms
        self.ver_probs = ver_probs

        self.clases_visibles = [self.labels.index(v) for v in self.labels]

        self.rpe = rpe
        self.rpi = rpi

        self.h5 = h5

self_ = v_ps(threshold = 0.25,
             batch_size = 16,
             dim_fil = 480, dim_col = 480,
             H = 13, W = 13, B = 3,
             learning_ratio = 1e-3,
             nms = True,
             ver_probs = False,
             rpe = '/home/sergio/Escritorio/Deep learning/0_Base_de_datos/COCO/labels_train',
             rpi = '/home/sergio/Escritorio/Deep learning/0_Base_de_datos/COCO/',
             h5 = 'mark1_COCO.h5')
             #rpe = 'C:/Users/Sergio.Coteron/Desktop/mark_s/base_de_datos/label_train',
             #rpi = 'C:/Users/Sergio.Coteron/Desktop/mark_s/base_de_datos/image_train/')
