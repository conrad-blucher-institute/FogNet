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

import numpy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from numpy import savez_compressed
import pandas 
import keras
import scipy
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
import scipy.io as sio 

import utils







#==========================================================================
YEAR_FOG_DIR_NAME = '.'
ALL_FOG_DIR_NAME = '..'
#DEFAULT_IMAGE_DIR_NAME = ('./Dataset/')
DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/')
#DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/2018/')
DEFAULT_NAMES_DIR_NAME = ('../Dataset/NAMES/')
DEFAULT_CUBES_DIR_NAME = ('../Dataset/INPUT/MinMax/')
DEFAULT_CUBES_12_DIR_NAME = ('../Dataset/INPUT/12Hours/')
DEFAULT_TARGET_DIR_NAME = ('../Dataset/TARGET')
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
TIME_CYCLE_FORMAT = '[0-9][0-9][0][0]'
HOUR_PREDICTION_FORMAT = '[0][0-9][0-9]'
#==========================================================================
# Reading the target first: 
CSVfile = '{0:s}/Target.csv'.format(DEFAULT_TARGET_DIR_NAME) 

'''year_information = {'2009':['20090101', '20091231'],
'2010':['20100101', '20101231'],
'2011':['20110101', '20111231'],
'2012':['20120101', '20121231'],
'2013':['20130101', '20131231'],
'2014':['20140101', '20141231'],
'2015':['20150101', '20151231'],
'2016':['20160101', '20161231'],
'2017':['20170101', '20171231'],
'2018':['20180101', '20181231'],
'2019':['20190101', '20191231'],
'2020':['20200101', '20201231']}'''
	



year_information = {
'2012':['20120101', '20121231'],
'2013':['20130101', '20131231'],
'2014':['20140101', '20141231'],
'2015':['20150101', '20151231'],
'2016':['20160101', '20161231'],
'2017':['20170101', '20171231'],
'2018':['20180101', '20181231'],
'2019':['20190101', '20191231'],
'2020':['20200101', '20201231']} 


sst_cube_names = {'2009':['NETCDF_SST_CUBE_2009.npz'],
'2010':['NETCDF_SST_CUBE_2010.npz'],
'2011':['NETCDF_SST_CUBE_2011.npz'],
'2012':['NETCDF_SST_CUBE_2012.npz'],
'2013':['NETCDF_SST_CUBE_2013.npz'],
'2014':['NETCDF_SST_CUBE_2014.npz'],
'2015':['NETCDF_SST_CUBE_2015.npz'],
'2016':['NETCDF_SST_CUBE_2016.npz'],
'2017':['NETCDF_SST_CUBE_2017.npz'],
'2018':['NETCDF_SST_CUBE_2018.npz'],
'2019':['NETCDF_SST_CUBE_2019.npz'],
'2020':['NETCDF_SST_CUBE_2020.npz']}



strategy = tensorflow.distribute.experimental.MultiWorkerMirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))       # just to show how many GPU are you using: 
with strategy.scope(): 
	#==========================================================================
	#========================= Cube Generating ================================
	#==========================================================================
	target = utils.reading_csv_target_file(CSVfile) 
	_ = utils.npz_cube_creator(year_information, target, DEFAULT_IMAGE_DIR_NAME)







#==========================================================================
#======================= upsample SST Maps ================================
#==========================================================================
#_ = utils.SST_384_cubes(sst_cube_names, DEFAULT_CUBES_DIR_NAME) 





'''nam =  DEFAULT_NAMES_DIR_NAME + 'NamFileNames2009.txt' 
# how to read a txt file: 
# open file and read the content in a list
training_nam_cubes_names = []
with open(nam, 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        training_nam_cubes_names.append(currentPlace)
#print(len(training_nam_cubes_names))
#print('The number of NAM training cubes: ', len(training_nam_cubes_names))

nam_concatcubes_names = utils.find_nam_cubes_name_hourpredict(training_nam_cubes_names, 
	hour_prediction_names = ['000', '003', '006'])
#print('The number of cubes of: ', len(nam_concatcubes_names))


normalized_cubes_dict = utils.scaler_many_cubes(nam_concatcubes_names, 
	utils.NETCDF_PREDICTOR_NAMES['NewOrder'])
#print(len(normalized_cubes_dict[0])) 
nam_concat_cubes = utils.concate_nam_cubes_files(normalized_cubes_dict, utils.NETCDF_PREDICTOR_NAMES['NewOrder']) 
#print(nam_concat_cubes.shape) 
nam_cube = utils.nam_cubes_dict(nam_concat_cubes) 
print("The trainig cube size of whole cube: ", nam_cube['predictor_matrix'].shape) 


sample_map = nam_concatcubes_names [0]
#print(sample_map) 


this_cube = utils.read_nam_maps(sample_map, utils.NETCDF_PREDICTOR_NAMES['NewOrder']) 
#print(this_cube['predictor_matrix'].shape) 
#print(this_cube['predictor_matrix'][0, 0:5,10, 0])

normalized_cube = utils.scaler_nam_maps(this_cube['predictor_matrix'], 
	utils.NETCDF_PREDICTOR_NAMES['NewOrder']) 

#print(normalized_cube['predictor_matrix'].shape) 
#print(normalized_cube['predictor_matrix'][0, 0:5,10, 0])'''