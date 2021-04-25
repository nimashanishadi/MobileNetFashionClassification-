from __future__ import print_function

from config import *
#from model import *
#from modelwithfreezall import *
from mobilenet_model import *


# Data Generation
def data_aug(train_path, validation_path, batch_size,batch_size_val, img_r, img_c):
  train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

  val_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

    
  train_batches = train_gen.flow_from_directory(train_path,
          target_size=(img_r, img_c),
          batch_size=batch_size,
          class_mode='categorical',
          shuffle=True)
          
  val_batches = val_gen.flow_from_directory(validation_path,
          target_size=(img_r, img_c),
          batch_size=batch_size_val,
          class_mode='categorical',
          shuffle=False)
  
  return train_batches, val_batches


    
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  elif (epoch > 10 and epoch < 30) :
    return 0.000001
  else:
    return 0.0000001
    

def fit_model(model, batches, val_batches, callbacks):

    print("started model training")
    history = model.fit(train_batches,
                                  steps_per_epoch = 209222//32,
                                  epochs = 50,
                                  validation_data= val_batches,
                                  validation_steps=40000//32,
                                  callbacks=callbacks,
                                  verbose=1,
                                  use_multiprocessing=True,
                                  #workers=8
                                  )
    print("evaluate_generator")
    score = model.evaluate_generator(val_batches,steps=len(val_batches.filenames)//batch_size_val, verbose=1)
    
    return model,history, score




if __name__ == '__main__':

    print("callback LearningRateScheduler")
    rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy",factor=0.9, patience=1,verbose=1)
    rlronp1=tf.keras.callbacks.ReduceLROnPlateau(monitor="top3_acc",factor=0.9, patience=1,verbose=1)
    rlronp2=tf.keras.callbacks.ReduceLROnPlateau(monitor="top5_acc",factor=0.9, patience=1,verbose=1)
    #estop=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=4,verbose=1,restore_best_weights=True)
    
    #callbacks=[rlronp, estop]
    callback=[rlronp,rlronp1,rlronp2]

    #callback  = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    model = create_model((224,224,3),50)
    train_batches, val_batches = data_aug(dataset_train_path, dataset_val_path, batch_size_train,batch_size_val, img_width, img_height)
    print(len(train_batches))
    model, history, score = fit_model(model, train_batches, val_batches, callbacks=[callback])
    print(round(model.optimizer.lr.numpy(), 5))
    print('Done......................')
    model.save('./model')
    model.save_weights("my_model_weights.h5")
    model.save_weights('./weights')