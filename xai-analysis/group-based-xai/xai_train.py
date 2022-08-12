import pandas as pd
import numpy
import os
import os.path 
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import errno
import glob
import time
import calendar
import json
import pickle
import random
import netCDF4
import copy
from numpy import savez_compressed
from optparse import OptionParser
from scipy.interpolate import (UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
from sklearn.metrics import confusion_matrix
import statistics
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model   
from tensorflow.keras.layers import Add, add, concatenate, Reshape, BatchNormalization, Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Conv3D, Activation, MaxPooling2D, MaxPooling3D, AveragePooling3D, ReLU, GlobalAveragePooling3D, multiply
 
from tensorflow.keras import regularizers 
from tensorflow.keras import optimizers 
from tensorflow.keras.optimizers import Adam, SGD 
from tensorflow.keras.callbacks import ModelCheckpoint


from scipy.io import loadmat
from tensorflow.keras.callbacks import EarlyStopping


'''import keras.backend.tensorflow_backend as tfback
def _get_available_gpus():
	if tfback._LOCAL_DEVICES is None:
	    devices = tensorflow.config.list_logical_devices()
	    tfback._LOCAL_DEVICES = [x.name for x in devices]
	return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
from keras.utils import multi_gpu_model'''
#from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
#import utils
#import FogNet
#import FogNetConfig

import sys
sys.path.append('../')
from src import utils, FogNet, FogNetConfig, cnn_evaluate, models, xai_engine

#import xai_engine
#import utils, FogNet, FogNetConfig, cnn_evaluate


DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/')
DEFAULT_TARGET_DIR_NAME = ('/data1/fog/fognn/Dataset/TARGET/')

DEFAULT_CUBES_24_DIR_NAME = ('/data1/fog/fognn/Dataset/24HOURS/INPUT/')
DEFAULT_TARGET_24_DIR_NAME = ('/data1/fog/fognn/Dataset/24HOURS/TARGET/')

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3
DEFAULT_RANDOM_LINE_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_RANDOM_LINE_WIDTH = 2

LEVELS_FOR_CONTOURS = numpy.linspace(0, 1, num=11, dtype=float)

FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

FONT_SIZE = 20
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=FONT_SIZE)
plt.rc('ytick', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=FONT_SIZE)

strategy = tensorflow.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

############################
# Setup input data rasters #
############################
# Generate data file paths
trainYearIdxs = [4, 5, 6, 7, 8]
valYearIdxs   = [0, 1, 2, 3]
testYearIdxs  = [9, 10, 11]

horizons = [6, 12, 24]
allYears = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

#strategy = tensorflow.distribute.experimental.MultiWorkerMirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#with strategy.scope(): 
    
nam_G1_template = "NETCDF_NAM_CUBE_{year}_PhG1_{horizon}.npz"
nam_G1_names = [nam_G1_template.format(year=year, horizon=horizons[2]) for year in allYears]

nam_G2_template = "NETCDF_NAM_CUBE_{year}_PhG2_{horizon}.npz"
nam_G2_names = [nam_G2_template.format(year=year, horizon=horizons[2]) for year in allYears]

nam_G3_template = "NETCDF_NAM_CUBE_{year}_PhG3_{horizon}.npz"
nam_G3_names = [nam_G3_template.format(year=year, horizon=horizons[2]) for year in allYears]

nam_G4_template = "NETCDF_NAM_CUBE_{year}_PhG4_{horizon}.npz"
nam_G4_names = [nam_G4_template.format(year=year, horizon=horizons[2]) for year in allYears]

mixed_file_template = "NETCDF_MIXED_CUBE_{year}_{horizon}.npz"
mixed_file_names = [mixed_file_template.format(year=year, horizon=horizons[2]) for year in allYears]

mur_file_template = "NETCDF_SST_CUBE_{year}.npz"
mur_file_names = [mur_file_template.format(year=year) for year in allYears]

targets_file_template = "target{year}_{horizon}.csv"
targets_file_names = [targets_file_template.format(year=year, horizon=horizons[2]) for year in allYears]


    # Read data cubes
training_list   = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, DEFAULT_CUBES_24_DIR_NAME, trainYearIdxs)

validation_list = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, DEFAULT_CUBES_24_DIR_NAME, valYearIdxs)

testing_list    = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, DEFAULT_CUBES_24_DIR_NAME, testYearIdxs)

target_class = utils.targets(
    targets_file_names, trainYearIdxs, valYearIdxs, testYearIdxs,
    DEFAULT_TARGET_24_DIR_NAME,
    0, # priority_calss: the last integer value is the class of target to predict: 0: is < 1600; 1: < 3200 and 2: < 6400
)
target_list = target_class.binary_target()

# Separate into train, test, validation
Training_targets = target_list[0]
print('training target shape:', Training_targets.shape)
ytrain = target_list[1]
print('training categorical target shape:', ytrain.shape)
Validation_targets = target_list[2]
print('validation target shape:', Validation_targets.shape)
yvalid = target_list[3]
print('validation categorical target shape:', yvalid.shape)
Testing_targets = target_list[4]
print('testing target shape:', Testing_targets.shape)
ytest = target_list[5]
print('testing categorical target shape:', ytest.shape)

#with strategy.scope():
    # Initialize
learningRate = 0.0009 # hyperparameters[key][0] 
wd           = 0.001  # hyperparameters[key][1] 
filters      = 24     # hyperparameters[key][2] 
dropout      = 0.3    # hyperparameters[key][3] 


cnn_file_name = '/data1/fog/FogNet/trained_model/single_gpu_weights.h5'
#cnn_file_name = '/data1/fog/fognn/FogNet/trained_model/weights.h5'
input_nam_shape       = Input((32, 32, 288, 1))
input_mur_shape       = Input((384, 384, 1, 1)) 
input_nam_G1_24_shape = Input((32, 32, 108, 1))
input_nam_G2_24_shape = Input((32, 32, 96, 1))
input_nam_G3_24_shape = Input((32, 32, 108, 1))
input_nam_G4_24_shape = Input((32, 32, 60, 1))
input_mixed_24_shape  = Input((32, 32, 12, 1))


C  = FogNet.FogNet(input_nam_G1_24_shape, 
               input_nam_G2_24_shape, 
               input_nam_G3_24_shape, 
               input_nam_G4_24_shape, 
               input_mixed_24_shape, 
               input_mur_shape, filters, dropout, 2)

cnn_model_object = C.BuildModel()
#cnn_model_object = multi_gpu_model(cnn_model_object, gpus=4)
#model.summary() 

cnn_model_object.load_weights(cnn_file_name)  

cnn_model_object.compile(optimizer=Adam(lr=learningRate, decay=wd),
      loss='categorical_crossentropy',
      metrics=['accuracy'])


PFI_channels = xai_engine.PFI_channels(cnn_model_object, testing_list, Testing_targets, n_repeats=5, random_state=42)
PFI_channels.to_csv('./PFICW.csv')


#PFI_groups = xai_engine.FIGW(cnn_model_object, testing_list, Testing_targets, xai_method = 'PFI', num_iter = 5, random_state=42)
#PFI_groups.to_csv('./PFIG3.csv')

#LossSHAP_groups = xai_engine.FIGW(cnn_model_object, testing_list, Testing_targets, xai_method = 'LossSHAP', num_iter = 5, random_state=42)
#LossSHAP_groups.to_csv('./LossSHAP5.csv')



