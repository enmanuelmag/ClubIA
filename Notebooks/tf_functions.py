import gc
import numpy as np
import random as rd
import pickle as pkl
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras import backend as k
from tensorflow.keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers

SEED = 42
#LEN_TF_RECORD = 11753
LEN_TF_RECORD = 4007
SUFIX = '_faces' if True else ''
PATH_VECTORS = f'./data/data_image_{LEN_TF_RECORD}{SUFIX}.pkl'

rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


class ReleaseMemory(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    k.clear_session()


class ViClassifier(tf.keras.Model):
  def __init__(self, num_classes, input_shape, units=1024, inner_layers=12, type_extractor='vgg'):
    assert type_extractor in ['vgg', 'inception', 'resnet']
    assert inner_layers >= 1
    assert len(input_shape) == 3
    assert units >= 64

    super(ViClassifier, self).__init__(name='ViClassifier')

    self.units = units
    self.in_layer = tf.keras.layers.Input(input_shape, name='input')

    if type_extractor == 'vgg':
      feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    elif type_extractor == 'inception':
      feature_extractor = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    elif type_extractor == 'resnet':
      feature_extractor = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    else:
      raise ValueError('type_extractor must be vgg, inception or resnet')

    self.feature_extractor = feature_extractor
    self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
    self.flatten = tf.keras.layers.Flatten()

    self.hidden_mlp = []
    for i in range(inner_layers):
      self.hidden_mlp.append(tf.keras.layers.Dense(units,activation=tf.nn.relu))
      self.hidden_mlp.append(tf.keras.layers.Dropout(0.5, seed=SEED))

    self.out_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

  def call(self, inputs, training=None, mask=None):
    x = self.feature_extractor(inputs, training=training)
    x = self.global_average_pooling(x)
    x = self.flatten(x, training=training)
    for layer in self.hidden_mlp:
      x = layer(x, training=training)
    return self.out_layer(x, training=training)

  def predict_class(self, x):
    return tf.argmax(self(x), axis=1)

  def vectorize(self, x, flatten=True):
    x = self.feature_extractor(x)
    x = self.global_average_pooling(x)
    if flatten:
      return self.flatten(x)
    return x


def parse_tfrecord_vec(tfrecord, size):
  x = tf.io.parse_single_example(tfrecord, {
    'image': tf.io.FixedLenFeature([], tf.string),
    'class_name': tf.io.FixedLenFeature([], tf.string),
  })
  x_train = tf.image.decode_jpeg(x['image'], channels=3)
  x_train = tf.image.resize(x_train, (size, size))
  x_train = preprocess_input(x_train, mode='tf')

  y_train = x['class_name']
  if y_train is None:
    y_train = ''

  return x_train, y_train

def load_tfrecord_dataset_vec(file_pattern, size):
  files = tf.data.Dataset.list_files(file_pattern)
  dataset = files.flat_map(tf.data.TFRecordDataset)
  return dataset.map(lambda x: parse_tfrecord_vec(x, size))

def parse_record_vec(combination):
  item_1, item_2 = combination
  img_1, label_1 = item_1
  img_2, label_2 = item_2
  return (img_1, img_2, label_1 == label_2)


def process_image_tf(image, size):
  print('Image byte', image)
  #image = tf.io.read_file(image_path)
  #image = tf.image.decode_jpeg(image, channels=3)
  image = Image.open(image).convert("RGB")
  #image = tf.image.decode_jpeg([ image ], channels=3)[0]
  image = tf.image.resize(image, (size, size))
  return preprocess_input(image, mode='tf')

