#===============================================================================================================
#====================================== All Packages from Tensorflow and Keras =================================
#===============================================================================================================
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
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
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
#===============================================================================================================
#====================================== All Basic Packages =====================================================
#===============================================================================================================
import numpy
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
import seaborn as sns
import pandas 
import scipy
from numpy import savez_compressed
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
import scipy.io as sio 
from optparse import OptionParser
#===============================================================================================================
#====================================== My own Packages ========================================================
#===============================================================================================================
import utils
import cnn_evaluate 
import FogNetConfig
import FogNet

#===============================================================================================================
#====================================== Parse options  =========================================================
#===============================================================================================================

parser = OptionParser()
# Project
parser.add_option("-n", "--name",
    help="Model name [default = %default].",
    default="test")

# Directories
parser.add_option("-d", "--directory",
    help="Fog dataset directory\n[default = %default].",
    default="/data1/fog/Dataset/")
parser.add_option("-o", "--output_directory",
    help="Output results directory [default = %default].")
parser.add_option(      "--force",
    help="Force overwrite of existing output directory [default = %default].",
    default=False, action="store_true")

# Data
parser.add_option("-t", "--time_horizon",
    help="Prediction time horizon [default = %default].",
    default="24")
parser.add_option(      "--train_years",
    help="Comma-delimited list of training years [default = %default].",
    default="2013,2014,2015,2016,2017")
parser.add_option(      "--val_years",
    help="Comma-delimited list of validation years [default = %default].",
    default="2009,2010,2011,2012")
parser.add_option(      "--test_years",
    help="Comma-delimited list of testing years [default = %default].",
    default="2018,2019,2020")

# Fog definitions
parser.add_option("-v", "--visibility_class",
    help="Visibility class [default = %default].",
    default=0)

# Hyperparameters
parser.add_option("-b", "--batch_size",
    help="Training batch size [default = %default].",
    default=32, type="int")
parser.add_option("-e", "--epochs",
    help="Training epochs [default = %default].",
    default=30, type="int")
parser.add_option(      "--learning_rate",
    help="Learning rate [default = %default].",
    default=0.0009, type="float")
parser.add_option(      "--weight_penalty",
    help="Weight penalty [default = %default].",
    default=0.001, type="float")
parser.add_option(      "--filters",
    help="Number of filters [default = %default].",
    default=24, type="int")
parser.add_option(      "--dropout",
    help="Droput rate [default = %default].",
    default=0.3, type="float")
(options, args) = parser.parse_args()

print("")
print("FogNet")
print("------")

# Model name
model_name = options.name
print("Model name: {}".format(model_name))

# Time horizon (i.e. 6, 12, 24)
horizon = options.time_horizon
print("Prediction horizon: {} hours".format(horizon))

# Visibility class
visibilityClasses = [1600, 3200, 6400]
try:
    visibilityClass = int(options.visibility_class)
    assert (visibilityClass >= 0 and visibilityClass < len(visibilityClasses))
except:
    print("Invalid visibility class {}. Expected integer in range [0, {}]".format(
        visibilityClass, len(visibilityClasses)))
    print("Exiting...")
    exit(1)
print("Visibility class {} --> {}".format(visibilityClass, visibilityClasses[visibilityClass]))

# Data directory
dataDir = options.directory
if not os.path.isdir(dataDir):
    print("Cannot find data directory: {}".format(dataDir))
    print("Exiting...")
    exit(1)
print("Data directory: {}".format(dataDir))

# Output directory
outDir = options.output_directory
force=options.force
if outDir is None:
    print("Must specify an output directory with `-o`")
    print("Exiting...")
    exit(1)
if os.path.isdir(outDir) and force==False:
    print("Output directory {} already exists! Use '--force' to overwrite.".format(outDir))
    print("Exiting...")
    exit(1)
if not os.path.isdir(outDir):
    os.makedirs(outDir)
print("Output directory: {}".format(outDir))

# Training, testing, validation years
trainYears = options.train_years.split(",")
print("Training years: {}".format(trainYears))
valYears = options.val_years.split(",")
print("Validation years: {}".format(valYears))
testYears = options.test_years.split(",")
print("Testing years: {}".format(testYears))

allYears = valYears + trainYears + testYears
start = 0
stop = len(valYears)
valYearIdxs = list(range(start, stop))
start = stop
stop = stop + len(trainYears)
trainYearIdxs = list(range(start, stop))
start = stop
stop = stop + len(testYears)
testYearIdxs = list(range(start, stop))

# Setup directories
cubeDir = "{}/{}HOURS/INPUT/".format(dataDir, horizon)
targetDir = "{}/{}HOURS/TARGET/".format(dataDir, horizon)

# Hyperparameters
batchSize = options.batch_size
epochs = options.epochs
learningRate = options.learning_rate
wd = options.weight_penalty
filters = options.filters 
dropout = options.dropout
print("Hyperparameters:")
print("    batch size: {}".format(batchSize))
print("    learning rate: {}".format(learningRate))
print("    weight penalty: {}".format(wd))
print("    number of filters: {}".format(filters))
print("    dropout rate: {}".format(dropout))

#===============================================================================================================
#================================= Input Cube Names ============================================================
#===============================================================================================================
nam_G1_template = "NETCDF_NAM_CUBE_{year}_PhG1_{horizon}.npz"
nam_G1_names = [nam_G1_template.format(year=year, horizon=horizon) for year in allYears]

nam_G2_template = "NETCDF_NAM_CUBE_{year}_PhG2_{horizon}.npz"
nam_G2_names = [nam_G1_template.format(year=year, horizon=horizon) for year in allYears]

nam_G3_template = "NETCDF_NAM_CUBE_{year}_PhG1_{horizon}.npz"
nam_G3_names = [nam_G1_template.format(year=year, horizon=horizon) for year in allYears]

nam_G4_template = "NETCDF_NAM_CUBE_{year}_PhG1_{horizon}.npz"
nam_G4_names = [nam_G1_template.format(year=year, horizon=horizon) for year in allYears]

mixed_file_template = "NETCDF_MIXED_CUBE_{year}_{horizon}.npz"
mixed_file_names = [mixed_file_template.format(year=year, horizon=horizon) for year in allYears]

mur_file_template = "NETCDF_SST_CUBE_{year}.npz"
mur_file_names = [mur_file_template.format(year=year) for year in allYears]

targets_file_template = "target{year}_{horizon}.csv"
targets_file_names = [targets_file_template.format(year=year, horizon=horizon) for year in allYears]

#===============================================================================================================
#====================================== Read Cube Files ========================================================
#===============================================================================================================
# If you are using more than one GPU you need to run this line to atke advantage of all the GPUs. 
#strategy = tensorflow.distribute.experimental.MultiWorkerMirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))       # just to show how many GPU are you using: 


#with strategy.scope(): 



  #shuff_order  = subdir_name + '_order' + '_report.txt' 

  #shuffled_idx = numpy.random.permutation(108)
  #print('Order  : ', shuffled_idx,  file=open(shuff_order, "a")) 

training_list   = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, cubeDir, trainYearIdxs)

validation_list = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, cubeDir, valYearIdxs)

testing_list    = utils.load_Cat_cube_data(nam_G1_names,
    nam_G2_names, nam_G3_names, nam_G4_names, mixed_file_names, mur_file_names, cubeDir, testYearIdxs)

target_class = utils.targets(
    targets_file_names, trainYearIdxs, valYearIdxs, testYearIdxs,
    targetDir,
    visibilityClass, # priority_calss: the last integer value is the class of target to predict: 0: is < 1600; 1: < 3200 and 2: < 6400
)
target_list = target_class.binary_target()

Training_targets = target_list[0]
print('training target shape:', Training_targets.shape)
#numpy.savetxt('training_c3.csv', Training_targets, delimiter=',', fmt='%d')

ytrain = target_list[1]
print('training categorical target shape:', ytrain.shape)

Validation_targets = target_list[2]
print('validation target shape:', Validation_targets.shape)
#numpy.savetxt('validation_c3.csv', Validation_targets, delimiter=',', fmt='%d')

yvalid = target_list[3]
print('validation categorical target shape:', yvalid.shape)

Testing_targets = target_list[4] 
print('testing target shape:', Testing_targets.shape)
#numpy.savetxt('testing_c3.csv', Testing_targets, delimiter=',', fmt='%d')

ytest = target_list[5]
print('testing categorical target shape:', ytest.shape)

  #====================================================================================
callbacks = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
  #callbacks = tensorflow.keras.callbacks.LearningRateScheduler(modelconfig.scheduler)

#for i in range(1,11):

N = 'run' + '11'
subdir_name = outDir + "/"  + N + '/' 
if not os.path.exists(subdir_name):
    os.makedirs(subdir_name) 

C  = FogNet.FogNet(
    Input(training_list[0].shape[1:]),
    Input(training_list[1].shape[1:]),
    Input(training_list[2].shape[1:]),
    Input(training_list[3].shape[1:]),
    Input(training_list[4].shape[1:]),
    Input(training_list[5].shape[1:]),
    filters, dropout, 2)

model = C.BuildModel() 
model = multi_gpu_model(model, gpus=4)
#model.load_weights('weights.h5')

model.compile(optimizer=Adam(lr=learningRate, decay=wd), 
      loss='categorical_crossentropy',
      metrics=['accuracy']) 
history = model.fit(x= training_list, y=ytrain,
  batch_size = batchSize, 
  epochs = epochs, 
  callbacks = [callbacks],
  validation_data = (validation_list, yvalid))

model.save(subdir_name + 'history.h5')
model.save_weights(subdir_name + 'weights.h5')

loss_name = subdir_name + 'loss.png'
_ = cnn_evaluate.plot_loss_function(history, loss_name)

#predict the output using predict function: 
column_names = ["C0_Prob", "C1_Prob"]
y_training_cat_prob     = model.predict(training_list)
VIS_Prob_TRAIN = pandas.DataFrame(y_training_cat_prob, columns = column_names)
#VIS_Prob_TRAIN['C0_Prob'] = y_training_cat_prob[:, 0]
#VIS_Prob_TRAIN['C1_Prob'] = y_training_cat_prob[:, 1] 
VIS_Prob_TRAIN.to_csv(subdir_name + 'VIS_Prob_TRAIN.csv')

y_validation_cat_prob   = model.predict(validation_list) 
VIS_Prob_VALID = pandas.DataFrame(y_validation_cat_prob, columns = column_names)
#VIS_Prob_VALID['C0_Prob'] = y_validation_cat_prob[:, 0]
#VIS_Prob_VALID['C1_Prob'] = y_validation_cat_prob[:, 1]
VIS_Prob_VALID.to_csv(subdir_name + 'VIS_Prob_VALID.csv')

y_testing_cat_prob       = model.predict(testing_list) 
VIS_Prob_TEST = pandas.DataFrame(y_testing_cat_prob, columns = column_names)
#VIS_Prob_TEST['C0_Prob'] = y_testing_cat_prob[:, 0]
#VIS_Prob_TEST['C1_Prob'] = y_testing_cat_prob[:, 1] 
VIS_Prob_TEST.to_csv(subdir_name + 'VIS_Prob_TEST.csv') 

Tr_name   = subdir_name + model_name + '_training' + '_0' + '_report.txt' 
val_name  = subdir_name + model_name + '_validation'+ '_0' + '_report.txt'
test_name = subdir_name + model_name + '_testing' + '_0' + '_report.txt' 

training_threshold   = cnn_evaluate.skilled_metrics(Training_targets, y_training_cat_prob, 'HSS', Tr_name)
accuray_list_validation = cnn_evaluate.skilled_metrics(Validation_targets, y_validation_cat_prob, 'HSS', val_name)
testing_threshold    = cnn_evaluate.skilled_metrics(Testing_targets, y_testing_cat_prob, 'HSS', test_name)   

test_accuracy_report = subdir_name + model_name + '_testing' + '_0' + '_accuracy_report.txt'
testing_accuracy = cnn_evaluate.confusion_cnn(Testing_targets, y_testing_cat_prob, training_threshold, test_accuracy_report)
