import os
import numpy as np
import glob

import tensorflow as tf
import keras.optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from tensorflow.keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras import applications


import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from tensorflow.keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector, BatchNormalization
from tensorflow.keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Convolution2D, ZeroPadding2D
from keras.applications.vgg16 import preprocess_input


from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random



from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG
import matplotlib.pyplot as plt
import tensorflow as tf
import functools

import numpy as np
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
from tensorflow.keras.callbacks import TensorBoard


from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import warnings

dataset_path='.'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

img_width = 224             
img_height = 224            
img_channel = 3

epochs = 100
batch_size_train = 32
batch_size_val = 32

predictions_class_weight=0.5
predictions_iou_weight=0.5
prediction_class_prob_threshold = 0.80
prediction_iou_threshold = 0.70
early_stopping_patience=500


input_image_width_threshold=500
input_image_height_threshold=500

optimizer='SGD'        

learn_rate=0.001
decay=0.0
momentum=0.0
activation='relu'
dropout_rate=0.5


warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import colored_traceback
#colored_traceback.add_hook(always=True)
#FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
