from __future__ import print_function

from config import *
import pathlib
import cv2

def init():

    global batch_size
    batch_size = batch_size_train
    logging.debug('batch_size {}'.format(batch_size))

    global class_names
    class_names = sorted(get_subdir_list(dataset_train_path))
    logging.debug('class_names {}'.format(class_names))

    global input_shape
    input_shape = (img_width, img_height, img_channel)
    logging.debug('input_shape {}'.format(input_shape))
    
    
def MultiHeadsAttModel(l=7*7, d=1024 , dv=64, dout=1024, nv = 16 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))

    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
    att = tf.einsum('baik,baij->bakj',q, k)/np.sqrt(dv) #batch matrix multiplication
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)
    out = tf.einsum('bajk,baik->baji',att, v)
    out = Reshape([l, d])(out)
    out = Add()([out, q1])

    out = Dense(dout, activation = "relu")(out)

    return  Model(inputs=[q1,k1,v1], outputs=out)   
    
    

def create_model(input_shape, output_classes):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))
    
    mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet')
  
    x = mobile.layers[-6].input
    
    if True:
        x = Reshape([7*7,1024])(x)
        att = MultiHeadsAttModel(l=7*7, d=1024 , dv=64, dout=1024, nv = 16 )
        x = att([x,x,x])
        x = Reshape([7,7,1024])(x)   
        x = BatchNormalization()(x)

    x = mobile.get_layer('global_average_pooling2d')(x)
    x = mobile.get_layer('reshape_1')(x)
    x = mobile.get_layer('dropout')(x)
    x = mobile.get_layer('conv_preds')(x)
    x = mobile.get_layer('reshape_2')(x)
    output = Dense(units=50, activation='softmax')(x)
    
    model = Model(inputs=mobile.input, outputs=output)
    
    for layer in model.layers[:-23]:
        layer.trainable = False
    
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc' 
    
    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    
    #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(
                  optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy',top3_acc,top5_acc]) 

    return model

    
 