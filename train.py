
import numpy as np
import os
from random import shuffle

import tensorflow as tf

from class_vic import self_
from auxiliar_train import leer_datos_text, programa_para_cargar_lote
from neuralnetwork import mark1, loss_function

"""
Clase que contiene las rutas de los txt, etiquetas de las imágenes.
"""
class solo_nombres:
    def __init__(self, _imageLabelNombre):
        self.imageLabelNombre = _imageLabelNombre
            
imageLabelNombre = leer_datos_text(ruta = self_.rpe)
shuffle(imageLabelNombre)
sn = solo_nombres(imageLabelNombre)

"""
Cargamos el modelo si existe o lo iniciamos de cero si no tenemos un .h5 en la ruta self_.h5.

Mostramos luego el resumen de la red neuronal.
"""
if os.path.exists(self_.h5):

    #model = tf.keras.models.load_model(self_.h5, custom_objects={'loss_function': loss_function})

    model, h_out = mark1(self_)
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.0001))
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))
    model.load_weights(self_.h5)
    
else:

    model, h_out = mark1(self_)
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))

print('')
print(model.summary())
print('')


"""
Clase para generar los lotes y entrenar con .fit_generator()
"""
class MY_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = sn.imageLabelNombre[idx * self.batch_size:(idx + 1) * self.batch_size]
        #self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        image_train = []
        _yTrue = []
        
        for name in batch_x:
            try:
                _imagen_train, def_yTrue = programa_para_cargar_lote(self_, [name])
            except:
                continue

            if _imagen_train == []:
                continue
            
            image_train.append(_imagen_train[0])
            _yTrue.append(def_yTrue[0])

        return np.array(image_train), np.array(_yTrue)

##### =================================================================================== #####
my_training_batch_generator = MY_Generator(sn.imageLabelNombre, None, self_.batch_size)
##### =================================================================================== #####

"""
Código absurdo porque sino no me deja entrenar.
"""
def preparar_unlote(image_filenames, batch_size):

    batch_x = image_filenames[:batch_size]
        
    image_train = []
    _yTrue = []
    
    for name in batch_x:
        try:
            _imagen_train, def_yTrue = programa_para_cargar_lote(self_, [name])
        except:
            continue

        if _imagen_train == []:
            continue
        
        image_train.append(_imagen_train[0])
        _yTrue.append(def_yTrue[0])

    return np.array(image_train), np.array(_yTrue)

x_train_lote, y_train_lote = preparar_unlote(sn.imageLabelNombre, self_.batch_size)
model.fit(x_train_lote, y_train_lote, verbose=1)

while True:
    try:
        model.fit_generator(generator=my_training_batch_generator,
                            #validation_data=validation_generator,
                            steps_per_epoch= int(len(sn.imageLabelNombre) / self_.batch_size),
                            epochs=1,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=3,
                            max_queue_size=10)

        print('')
        print(' ===== salvando modelo =====')
        print('')
                
        tf.keras.models.save_model(model, self_.h5)

        shuffle(sn.imageLabelNombre)

    except:break

