
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical


from keras.models import Model
from keras.layers import concatenate
from keras.layers import Add, add, Reshape, BatchNormalization, Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, Conv3D, Activation, MaxPooling2D, MaxPooling3D, AveragePooling3D, ReLU, GlobalAveragePooling3D, multiply, PReLU
from keras.layers.advanced_activations import LeakyReLU 
from keras import optimizers 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint


from keras.utils import np_utils
from operator import truediv
from scipy.io import loadmat
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import keras.backend as K
from keras.callbacks import Callback 

import utils
import json
import math 

#======================================================================================================
def SpectralAttentionBlock(input, filters):
    # Attention Block 
    # Maxpooling
    maxpool = MaxPooling3D(pool_size=(32, 32, 1))(input)
    maxpool_Shape = maxpool.shape
    maxpool = Reshape((1, maxpool_Shape[4]))(maxpool)
    #print("The size of spectral maxpooling: ", maxpool._keras_shape)

    # Avgpooling
    avgpool = AveragePooling3D(pool_size=(32, 32, 1))(input)
    avgpool_Shape = avgpool.shape
    #print(avgpool_Shape) 
    avgpool = Reshape((1, avgpool_Shape[4]))(avgpool)
    #print("The size of spectral avgpooling: ", avgpool.shape)

    ElementWiseSum = add([maxpool, avgpool])
    ElementWiseSum = Dense(filters, activation="sigmoid")(ElementWiseSum)
    #print(ElementWiseSum.shape)
    multensor = multiply([ElementWiseSum, input]) 
    #print("The size of spectral attention tensor: ", multensor.shape)

    return multensor



def SpatialAttentionBlock(input):
    # Attention Block 
    # Maxpooling
    maxpool = MaxPooling3D(pool_size=(3, 3, 1), strides=(1,1,1), padding= 'same')(input)
    #print("The size of spatial maxpooling: ", maxpool._keras_shape)

    # Avgpooling
    avgpool = AveragePooling3D(pool_size=(3, 3, 1), strides=(1,1,1), padding= 'same')(input)
    #print("The size of spatial avgpooling: ",avgpool._keras_shape)

    Concat = concatenate([maxpool , avgpool], axis = 3)
    Dense = Conv3D(filters=1, kernel_size=(1, 1, 2))(Concat) 
    #print("The size of spatial attention Conv: ", Dense._keras_shape)

    multensor = multiply([Dense, input]) 
    #print("The size of spatial attention tensor: ", multensor._keras_shape)

    return multensor

def SpectralDenseFactor(inputs):
    h_1 = BatchNormalization()(inputs)
    h_1 = Conv3D(filters=12, kernel_size=(1, 1, 7), padding = 'same')(h_1) 
    output = PReLU()(h_1)
    return output 

def SpectralDenseBlock(inputs):
    concatenated_inputs = inputs
    for i in range(3):
        x = SpectralDenseFactor(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=4)
    return concatenated_inputs

def SpectralDenseReduction(inputs):
    h_1 = BatchNormalization()(inputs)
    h_1 = Conv3D(filters=12, kernel_size=(1, 1, 9), padding = 'same')(h_1) 
    output = PReLU()(h_1)
    return output 

def SpectralDenseBlockR(inputs):
    concatenated_inputs = inputs
    for i in range(3):
        x = SpectralDenseReduction(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=4)
    return concatenated_inputs

def SpatialDenseFactor(inputs):
    #h_1 = BatchNormalization()(inputs)
    h_1 = Conv3D(filters=12, kernel_size=(3, 3, 1), padding = 'same')(inputs) 
    output = PReLU()(h_1)
    return output

def SpatialDenseBlock(inputs):
    concatenated_inputs = inputs
    for i in range(3):
        x = SpatialDenseFactor(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=4)
    return concatenated_inputs


def SSTDilationBlock(input_map):
  input_map_norm = BatchNormalization()(input_map)
  Conv1_1 = Conv3D(filters = 16, kernel_size=(3, 3, 1),  dilation_rate=(1, 1, 1),
    padding ='same' )(input_map_norm)
  Conv1_1 = PReLU()(Conv1_1)  

  Conv1_2 = Conv3D(filters = 16, kernel_size=(3, 3, 1),  dilation_rate=(3, 3, 1),
    padding ='same')(input_map_norm)
  Conv1_2 = PReLU()(Conv1_2) 

  Conv1_3 = Conv3D(filters = 16, kernel_size=(3, 3, 1),  dilation_rate=(5, 5, 1),
    padding ='same')(input_map_norm)
  Conv1_3 = PReLU()(Conv1_3)
  Conv1 = concatenate([Conv1_1, Conv1_2, Conv1_3], axis = -1)

  return Conv1

def NAMDilationBlock(input_map):
  input_map_norm = BatchNormalization()(input_map)
  NConv1_1 = Conv3D(filters = 16, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1),
                               padding ='same')(input_map_norm)
  NConv1_1 = PReLU()(NConv1_1) 
  NConv1_2 = Conv3D(filters = 16, kernel_size=(3, 3, 3), dilation_rate=(3, 3, 3),
                               padding ='same')(input_map_norm)
  NConv1_2 = PReLU()(NConv1_2) 
  NConv1_3 = Conv3D(filters = 16, kernel_size=(3, 3, 3), dilation_rate=(5, 5, 5),
                               padding ='same')(input_map_norm)
  NConv1_3 = PReLU()(NConv1_3) 
  NConv1 = concatenate([NConv1_1, NConv1_2, NConv1_3], axis = -1)

  return NConv1
