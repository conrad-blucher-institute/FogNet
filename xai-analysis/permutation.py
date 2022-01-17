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


'''
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.models import Input, Model   
from keras.layers import Add, add, concatenate, Reshape, BatchNormalization, Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv3D, Activation, MaxPooling2D, MaxPooling3D, AveragePooling3D, ReLU, GlobalAveragePooling3D, multiply
from keras.layers.advanced_activations import LeakyReLU 
from keras import regularizers 
from keras import optimizers 
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from scipy.io import loadmat
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as tfback
#print("tf.version is", tf.version)
#print("tf.keras.version is:", tf.keras.version)

def _get_available_gpus():
	if tfback._LOCAL_DEVICES is None:
	    devices = tensorflow.config.list_logical_devices()
	    tfback._LOCAL_DEVICES = [x.name for x in devices]
	return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import utils
import FogNet
import FogNetConfig


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


# Initialize
learningRate = 0.0009 # hyperparameters[key][0] 
wd           = 0.001  # hyperparameters[key][1] 
filters      = 24     # hyperparameters[key][2] 
dropout      = 0.3    # hyperparameters[key][3] 
cnn_file_name = '/data1/fog/fognn/FogNet/trained_model/single_gpu_weights.h5'

C  = FogNet.FogNet(
    Input(training_list[0].shape[1:]),
    Input(training_list[1].shape[1:]),
    Input(training_list[2].shape[1:]),
    Input(training_list[3].shape[1:]),
    Input(training_list[4].shape[1:]),
    Input(training_list[5].shape[1:]),
    filters, dropout, 2)
cnn_model_object = C.BuildModel()
#model.summary() 

cnn_model_object.load_weights(cnn_file_name)  

cnn_model_object.compile(optimizer=Adam(lr=learningRate, decay=wd),
      loss='categorical_crossentropy',
      metrics=['accuracy'])



print("I am Here!")



from sklearn.metrics import confusion_matrix
def test_eval(y, ypred, th = None): 
    length = len(ypred) 
    ypred_ = [0]*length

    for i in range(length):
        prob = ypred[i, 1] 
        if prob > th:
            ypred_[i] = 1
        else:
            ypred_[i] = 0
    ypred_ = numpy.array(ypred_)
    tn, fp, fn, tp = confusion_matrix(y, ypred_).ravel()
    a = tn     # Hit
    b = fn      # false alarm
    c = fp      # miss
    d = tp    # correct rejection 

    POD = a/(a+c)
    F   = b/(b+d)
    FAR  = b/(a+b)
    CSI = a/(a+b+c)
    PSS = ((a*d)-(b*c))/((b+d)*(a+c))
    HSS = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
    ORSS = ((a*d)-(b*c))/((a*d)+(b*c))
    CSS = ((a*d)-(b*c))/((a+b)*(c+d))

    #print('POD  : ', POD) 
    #print('F    : ', F)
    #print('FAR  : ', FAR)
    #print('CSI  : ', CSI)
    #print('PSS  : ', PSS)
    #print('HSS  : ', HSS)
    #print('ORSS : ', ORSS)
    #print('CSS  : ', CSS)
    return [POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]

def permutation_importance_(model_object, input_data, input_target, n_repeats=None, random_state=42): 
    
    #df = pd.DataFrame(columns = ['Feature', 'POD_mean', 'POD_std', 'F_mean', 'F_std','FAR_mean', 'FAR_std','CSI_mean', 'CSI_std','PSS_mean', 'PSS_std', 'HSS_mean', 'HSS_std', 'ORSS_mean', 'ORSS_std', 'CSS_mean', 'CSS_std'])
    #df = pd.DataFrame(columns = ['Feature', 'POD_mean', 'F_mean','FAR_mean', 'CSI_mean', 'PSS_mean', 'HSS_mean', 'ORSS_mean','CSS_mean'])
    df = pd.DataFrame(columns = ['Feature', 'HSS_mean', 'HSS_std'])
    this_hss= []
    fnames = []
    n_groups   = len(input_data)
    for g in range(n_groups): 
    #g = 3
        if g ==0:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G1']
        elif g ==1:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G2']
        elif g ==2:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G3']
        elif g ==3:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G4']
        elif g ==4:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Mixed']
        elif g ==5:
            GNames = utils.NETCDF_PREDICTOR_NAMES['SST']        
        n_features = input_data[g].shape[3]

        for f in range(n_features):
            for i in range(n_repeats): 
                #print(f"We are in group {g}, with {n_features} features")
                
                input_data_copy = copy.deepcopy(input_data)

                numpy.random.seed(random_state)
                permuted_map = numpy.random.permutation(input_data_copy[g][:,:,:,f,:]) 
                input_data_copy[g][:,:,:,f,:] = permuted_map

                y_testing_cat_prob = model_object.predict(input_data_copy) 
                metric_list = test_eval(input_target, y_testing_cat_prob, th = 0.193)

                #this_pod.append(metric_list[0])
                #this_f.append(metric_list[1])
                #this_far.append(metric_list[2])
                #this_csi.append(metric_list[3])
                #this_pss.append(metric_list[4])
                this_hss.append(metric_list[5])
                #this_orss.append(metric_list[6])
                #this_css.append(metric_list[7])

                feature_name     = GNames[int(numpy.floor(f/4))]
                fnames.append(feature_name)

                #print(f"{feature_name}: HSS Mean = {this_hss}|") 
                
                
    #print(f"The calculation for feature {feature_name} is done!")
    df['Feature']    = fnames
    #df['POD_mean']   = this_pod
    #df['POD_std']    = numpy.std(this_pod)
    #df['F_mean']     = this_f
    #df['F_std']      = numpy.std(this_f)
   # df['FAR_mean']   = this_far
    #df['FAR_std']    = numpy.std(this_far)
    #df['CSI_mean']   = this_csi
    #df['CSI_std']    = numpy.std(this_csi)
    #df['PSS_mean']   = this_pss
    #df['PSS_std']    = numpy.std(this_pss)
    df['HSS_mean']   = this_hss
    df['HSS_std']    = numpy.std(this_hss)
    #df['ORSS_mean']  = this_orss
    #df['ORSS_std']   = numpy.std(this_orss)
    #df['CSS_mean']   = this_css
    #df['CSS_std']    = numpy.std(this_css)
    
    return df

results = permutation_importance_(cnn_model_object, testing_list, Testing_targets, n_repeats=2, random_state=42)
results.to_csv('./NewR.csv')