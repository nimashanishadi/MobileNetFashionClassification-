from __future__ import print_function
from config import *
from train import *

from keras.preprocessing import image

predicted_classes = []

def test_gen(test_path, batch_size, img_r, img_c):
  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=(224, 224),
                                                    color_mode="rgb",
                                                    shuffle = False,
                                                    class_mode='categorical',
                                                    batch_size=32)
  filenames = test_generator.filenames
  nb_samples = len(filenames)
  return test_generator, nb_samples
  


def predict_model(model,test_batches, nb_samples): 
  predict = model.predict_generator(test_batches,steps = np.ceil(nb_samples//32+1), verbose=1)
  loss, acc = model.evaluate_generator(test_batches, steps=np.ceil(nb_samples//32+1), verbose=1)
  return predict, loss, acc
  
  
  
  
if __name__ == '__main__':
  model = keras.models.load_model('./model')
  test_batches, nb_samples = test_gen(dataset_test_path, 32, img_width, img_height)
  predict, loss, acc = predict_model(model,test_batches, nb_samples)
  print(predict)
  print(acc)
  print(loss)
 





















