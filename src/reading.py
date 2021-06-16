
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.version is", tf.version)
#print("tf.keras.version is:", tf.keras.version)

def _get_available_gpus():
	"""Get a list of available gpu devices (formatted as strings).

	# Returns
	    A list of available GPU devices.
	"""
	#global _LOCAL_DEVICES
	if tfback._LOCAL_DEVICES is None:
	    devices = tf.config.list_logical_devices()
	    tfback._LOCAL_DEVICES = [x.name for x in devices]
	return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus



import numpy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
#import tensorflow as tf
#import keras.backend as K
import matplotlib.pyplot as pyplot
import copy
import errno
import random
import glob
import os.path
import time
import calendar
import json
import pickle
import netCDF4
import seaborn as sns
import numpy
import pandas 
import keras
import scipy
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
#import sklearn.metrics
import scipy.io as sio 

FIG_DEFULT_SIZE = (12, 10)

YEAR_FOG_DIR_NAME = '.'
ALL_FOG_DIR_NAME = '..'
#DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/')
DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/2020')
DEFAULT_TARGET_DIR_NAME = ('./Dataset/TARGET/')
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
TIME_CYCLE_FORMAT = '[0-9][0-9][0][0]'
HOUR_PREDICTION_FORMAT = '[0][0-9][0-9]'


# In[2]:


import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model  
from keras.layers import concatenate, Conv2D, Conv3D, ReLU, Flatten, Dense, Activation, Reshape, BatchNormalization, MaxPooling2D, MaxPooling3D, Dropout, Input, GlobalAveragePooling3D
from keras.layers.advanced_activations import LeakyReLU 
from keras import optimizers 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from scipy.io import loadmat
from keras.utils import to_categorical
from numpy import load 
from numpy import savez_compressed 
from sys import getsizeof


import utils
import MurUtils
import roc_curves
import performance_diagrams
import keras_metrics 
import cnn_evaluate 
import attributes_diagrams
import models
#import datagenerator




#2018_4Cycles_3Classes_time_series
#3Classes_Target_NAM
CSVfile = '{0:s}/Target.csv'.format(DEFAULT_TARGET_DIR_NAME) 
data = pandas.read_csv(CSVfile, header=0, sep=',') 

for i in range(len(data)):
    namesplit = os.path.split(data['Name'][i])[-1]
    year = namesplit.replace(namesplit[4:], '')
    month = namesplit.replace(namesplit[0:4], '').replace(namesplit[6:], '')
    day = namesplit.replace(namesplit[0:6], '').replace(namesplit[8:], '')
    timecycle = namesplit.replace(namesplit[0:9], '')

data['Year'] = year
data['Month']= month
data['Day']= day
data['TimeCycle'] = timecycle
data



print("Print the days which the input cube is lost!")
nam_cubes_names_2020, mur_cubes_names_2020, targets_2020 = utils.find_map_name_date(
    first_date_string='20090101', last_date_string='20200530', target = data, image_dir_name= DEFAULT_IMAGE_DIR_NAME)
print("===============================================================")
print('The number of NAM training cubes: ', len(nam_cubes_names_2020))
print('The number of MUR training cubes: ', len(mur_cubes_names_2020))



# save the test file into txt file:
with open('NamFileNames2020.txt', 'w') as filehandle:
    for listitem in nam_cubes_names_2020:
        filehandle.write('%s\n' % listitem)

        # save the test file into txt file:
with open('MurfileNames2020.txt', 'w') as filehandle:
    for listitem in mur_cubes_names_2020:
        filehandle.write('%s\n' % listitem) 

targets_2020.to_csv('target2020.csv')