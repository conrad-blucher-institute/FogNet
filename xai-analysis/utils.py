"""Helper methods for XAI."""

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
import numpy
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas
#from sklearn.metrics import confusion_matrix


#import keras

import tensorflow 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam


tensorflow.compat.v1.disable_eager_execution()

#from shap.plots import colors
#cmap_ = colors.red_transparent_blue


# Plotting constants.
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10
FIGURE_RESOLUTION_DPI = 300

BAR_GRAPH_FACE_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
BAR_GRAPH_EDGE_COLOUR = numpy.full(3, 0.)
BAR_GRAPH_EDGE_WIDTH = 2.

SALIENCY_COLOUR_MAP_OBJECT = plt.cm.Greys

FONT_SIZE = 20
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=FONT_SIZE)
plt.rc('ytick', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=FONT_SIZE)

THIS_COLOUR_LIST = [
    numpy.array([4, 233, 231]), numpy.array([1, 159, 244]),
    numpy.array([3, 0, 244]), numpy.array([2, 253, 2]),
    numpy.array([1, 197, 1]), numpy.array([0, 142, 0]),
    numpy.array([253, 248, 2]), numpy.array([229, 188, 0]),
    numpy.array([253, 149, 0]), numpy.array([253, 0, 0]),
    numpy.array([212, 0, 0]), numpy.array([188, 0, 0]),
    numpy.array([248, 0, 253]), numpy.array([152, 84, 198])
]

for p in range(len(THIS_COLOUR_LIST)):
    THIS_COLOUR_LIST[p] = THIS_COLOUR_LIST[p].astype(float) / 255

REFL_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(THIS_COLOUR_LIST)
REFL_COLOUR_MAP_OBJECT.set_under(numpy.ones(3))



THESE_COLOUR_BOUNDS = numpy.array(
    [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
REFL_COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    THESE_COLOUR_BOUNDS, REFL_COLOUR_MAP_OBJECT.N)

# Machine-learning constants.
L1_WEIGHT = 0.
L2_WEIGHT = 0.001
NUM_PREDICTORS_TO_FIRST_NUM_FILTERS = 8
NUM_CONV_LAYER_SETS = 2
NUM_CONV_LAYERS_PER_SET = 2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
CONV_LAYER_DROPOUT_FRACTION = None
USE_BATCH_NORMALIZATION = True
SLOPE_FOR_RELU = 0.2
NUM_POOLING_ROWS = 2
NUM_POOLING_COLUMNS = 2
NUM_DENSE_LAYERS = 3
DENSE_LAYER_DROPOUT_FRACTION = 0.5

NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5

MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.005
NUM_EPOCHS_FOR_EARLY_STOPPING = 5




DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = 0.01

# Misc constants.
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY
METRES_PER_SECOND_TO_KT = 3.6 / 1.852


# Variable names.
NETCDF_X = 'x'
NETCDF_Y = 'y'
MUR_LATITUDE = 'lat'
MUR_LONGITUDE = 'lon'
NETCDF_LATITUDE = 'latitude'
NETCDF_LONGITUDE = 'longitude'
NETCDF_TIME = 'time'
NETCDF_UGRD_10m   = 'UGRD_10maboveground'
#NETCDF_UGRD_1000mb= 'UGRD_1000mb'
NETCDF_UGRD_975mb = 'UGRD_975mb'
NETCDF_UGRD_950mb = 'UGRD_950mb'
NETCDF_UGRD_925mb = 'UGRD_925mb'
NETCDF_UGRD_900mb = 'UGRD_900mb'
NETCDF_UGRD_875mb = 'UGRD_875mb'
NETCDF_UGRD_850mb = 'UGRD_850mb'
NETCDF_UGRD_825mb = 'UGRD_825mb'
NETCDF_UGRD_800mb = 'UGRD_800mb'
NETCDF_UGRD_775mb = 'UGRD_775mb'
NETCDF_UGRD_750mb = 'UGRD_750mb'
NETCDF_UGRD_725mb = 'UGRD_725mb'
NETCDF_UGRD_700mb = 'UGRD_700mb'
NETCDF_VGRD_10m   = 'VGRD_10maboveground'
#NETCDF_VGRD_1000mb= 'VGRD_1000mb'
NETCDF_VGRD_975mb = 'VGRD_975mb'
NETCDF_VGRD_950mb = 'VGRD_950mb'
NETCDF_VGRD_925mb = 'VGRD_925mb'
NETCDF_VGRD_900mb = 'VGRD_900mb'
NETCDF_VGRD_875mb = 'VGRD_875mb'
NETCDF_VGRD_850mb = 'VGRD_850mb'
NETCDF_VGRD_825mb = 'VGRD_825mb'
NETCDF_VGRD_800mb = 'VGRD_800mb'
NETCDF_VGRD_775mb = 'VGRD_775mb'
NETCDF_VGRD_750mb = 'VGRD_750mb'
NETCDF_VGRD_725mb = 'VGRD_725mb'
NETCDF_VGRD_700mb = 'VGRD_700mb'
#NETCDF_VVEL_1000mb= 'VVEL_1000mb'
NETCDF_VVEL_975mb = 'VVEL_975mb'
NETCDF_VVEL_950mb = 'VVEL_950mb'
NETCDF_VVEL_925mb = 'VVEL_925mb'
NETCDF_VVEL_900mb = 'VVEL_900mb'
NETCDF_VVEL_875mb = 'VVEL_875mb'
NETCDF_VVEL_850mb = 'VVEL_850mb'
NETCDF_VVEL_825mb = 'VVEL_825mb'
NETCDF_VVEL_800mb = 'VVEL_800mb'
NETCDF_VVEL_775mb = 'VVEL_775mb'
NETCDF_VVEL_750mb = 'VVEL_750mb'
NETCDF_VVEL_725mb = 'VVEL_725mb'
NETCDF_VVEL_700mb = 'VVEL_700mb'
#NETCDF_TKE_1000mb = 'TKE_1000mb'
NETCDF_TKE_975mb = 'TKE_975mb'
NETCDF_TKE_950mb = 'TKE_950mb'
NETCDF_TKE_925mb = 'TKE_925mb'
NETCDF_TKE_900mb = 'TKE_900mb'
NETCDF_TKE_875mb = 'TKE_875mb'
NETCDF_TKE_850mb = 'TKE_850mb'
NETCDF_TKE_825mb = 'TKE_825mb'
NETCDF_TKE_800mb = 'TKE_800mb'
NETCDF_TKE_775mb = 'TKE_775mb'
NETCDF_TKE_750mb = 'TKE_750mb'
NETCDF_TKE_725mb = 'TKE_725mb'
NETCDF_TKE_700mb = 'TKE_700mb'
NETCDF_TMP_SFC  = 'TMP_surface'
NETCDF_TMP_2m    = 'TMP_2maboveground'
#NETCDF_TMP_1000mb= 'TMP_1000mb'
NETCDF_TMP_975mb = 'TMP_975mb'
NETCDF_TMP_950mb = 'TMP_950mb'
NETCDF_TMP_925mb = 'TMP_925mb'
NETCDF_TMP_900mb = 'TMP_900mb'
NETCDF_TMP_875mb = 'TMP_875mb'
NETCDF_TMP_850mb = 'TMP_850mb'
NETCDF_TMP_825mb = 'TMP_825mb'
NETCDF_TMP_800mb = 'TMP_800mb'
NETCDF_TMP_775mb = 'TMP_775mb'
NETCDF_TMP_750mb = 'TMP_750mb'
NETCDF_TMP_725mb = 'TMP_725mb'
NETCDF_TMP_700mb = 'TMP_700mb'
#NETCDF_RH_1000mb = 'RH_1000mb'
NETCDF_RH_975mb  = 'RH_975mb'
NETCDF_RH_950mb  = 'RH_950mb'
NETCDF_RH_925mb  = 'RH_925mb'
NETCDF_RH_900mb  = 'RH_900mb'
NETCDF_RH_875mb  = 'RH_875mb'
NETCDF_RH_850mb  = 'RH_850mb'
NETCDF_RH_825mb  = 'RH_825mb'
NETCDF_RH_800mb  = 'RH_800mb'
NETCDF_RH_775mb  = 'RH_775mb'
NETCDF_RH_750mb  = 'RH_750mb'
NETCDF_RH_725mb  = 'RH_725mb'
NETCDF_RH_700mb  = 'RH_700mb'
NETCDF_DPT_2m = 'DPT_2maboveground'
NETCDF_FRICV = 'FRICV_surface'
NETCDF_VIS = 'VIS_surface'
NETCDF_RH_2m     = 'RH_2maboveground'
#
NETCDF_Q975 = 'Q_975mb'
NETCDF_Q950 = 'Q_950mb'
NETCDF_Q925 = 'Q_925mb'
NETCDF_Q900 = 'Q_900mb'
NETCDF_Q875 = 'Q_875mb'
NETCDF_Q850 = 'Q_850mb'
NETCDF_Q825 = 'Q_825mb'
NETCDF_Q800 = 'Q_800mb'
NETCDF_Q775 = 'Q_775mb'
NETCDF_Q750 = 'Q_750mb'
NETCDF_Q725 = 'Q_725mb'
NETCDF_Q700 = 'Q_700mb'
NETCDF_Q = 'Q_surface'
#NETCDF_DQDZ1000SFC = 'DQDZ1000SFC'
NETCDF_DQDZ975SFC  = 'DQDZ975SFC'
NETCDF_DQDZ950975  = 'DQDZ950975'
NETCDF_DQDZ925950  = 'DQDZ925950'
NETCDF_DQDZ900925  = 'DQDZ900925'
NETCDF_DQDZ875900  = 'DQDZ875900'
NETCDF_DQDZ850875  = 'DQDZ850875'
NETCDF_DQDZ825850  = 'DQDZ825850'
NETCDF_DQDZ800825  = 'DQDZ800825'
NETCDF_DQDZ775800  = 'DQDZ775800'
NETCDF_DQDZ750775  = 'DQDZ750775'
NETCDF_DQDZ725750  = 'DQDZ725750'
NETCDF_DQDZ700725  = 'DQDZ700725'
NETCDF_LCLT = 'LCLT'
NETCDF_DateVal = 'DateVal'

#+++++++++++++++
NETCDF_SST = 'analysed_sst'
NETCDF_TMPDPT = 'TMP-DPT'
NETCDF_TMPSST = 'TMP-SST'
NETCDF_DPTSST = 'DPT-SST'






NETCDF_PREDICTOR_NAMES = {
    'Physical_G1':[NETCDF_FRICV, NETCDF_UGRD_10m, NETCDF_UGRD_975mb, NETCDF_UGRD_950mb, NETCDF_UGRD_925mb, NETCDF_UGRD_900mb,
    NETCDF_UGRD_875mb, NETCDF_UGRD_850mb, NETCDF_UGRD_825mb, NETCDF_UGRD_800mb, NETCDF_UGRD_775mb,  NETCDF_UGRD_750mb,
    NETCDF_UGRD_725mb, NETCDF_UGRD_700mb, NETCDF_VGRD_10m, NETCDF_VGRD_975mb, NETCDF_VGRD_950mb, NETCDF_VGRD_925mb,
    NETCDF_VGRD_900mb, NETCDF_VGRD_875mb, NETCDF_VGRD_850mb, NETCDF_VGRD_825mb, NETCDF_VGRD_800mb, NETCDF_VGRD_775mb, NETCDF_VGRD_750mb,
    NETCDF_VGRD_725mb, NETCDF_VGRD_700mb],
    'Physical_G2':[NETCDF_TKE_975mb, NETCDF_TKE_950mb, NETCDF_TKE_925mb, NETCDF_TKE_900mb, NETCDF_TKE_875mb, NETCDF_TKE_850mb, NETCDF_TKE_825mb,
    NETCDF_TKE_800mb, NETCDF_TKE_775mb, NETCDF_TKE_750mb, NETCDF_TKE_725mb, NETCDF_TKE_700mb, NETCDF_Q975, NETCDF_Q950, NETCDF_Q925, NETCDF_Q900,
    NETCDF_Q875, NETCDF_Q850, NETCDF_Q825, NETCDF_Q800, NETCDF_Q775,NETCDF_Q750, NETCDF_Q725, NETCDF_Q700],
    'Physical_G3':[NETCDF_TMP_2m, NETCDF_TMP_975mb, NETCDF_TMP_950mb, NETCDF_TMP_925mb, NETCDF_TMP_900mb, NETCDF_TMP_875mb, NETCDF_TMP_850mb,
    NETCDF_TMP_825mb, NETCDF_TMP_800mb, NETCDF_TMP_775mb, NETCDF_TMP_750mb, NETCDF_TMP_725mb, NETCDF_TMP_700mb, NETCDF_DPT_2m, NETCDF_RH_2m,
    NETCDF_RH_975mb, NETCDF_RH_950mb, NETCDF_RH_925mb,NETCDF_RH_900mb, NETCDF_RH_875mb, NETCDF_RH_850mb, NETCDF_RH_825mb, NETCDF_RH_800mb,
    NETCDF_RH_775mb, NETCDF_RH_750mb, NETCDF_RH_725mb, NETCDF_RH_700mb],
    'Physical_G4':[NETCDF_Q, NETCDF_LCLT, NETCDF_VIS,
    NETCDF_VVEL_975mb, NETCDF_VVEL_950mb, NETCDF_VVEL_925mb, NETCDF_VVEL_900mb, NETCDF_VVEL_875mb, NETCDF_VVEL_850mb, NETCDF_VVEL_825mb,
    NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb],
    'SST': [NETCDF_SST],
    'Mixed': [NETCDF_SST, NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST]
    }





#####################################################################################################################


def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """
    #return keras.models.load_model(
    #    hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT)
    #return keras.models.load_model(hdf5_file_name) 
    return keras.model.load_weights(hdf5_file_name)


def load_cube(name):
    load_yeras = numpy.load(name)
    load_yeras = load_yeras['arr_0']
    return load_yeras

def concat_cat_npz_cube_files(nam_G1_names, nam_G2_names, nam_G3_names,nam_G4_names, mixed_file_names, mur_file_names, root, years):

    NAM_CUBE_G1   = None
    NAM_CUBE_G2   = None
    NAM_CUBE_G3   = None
    NAM_CUBE_G4   = None
    SST_CUBE      = None
    MIXED_CUBE    = None


    for i in years:
        nam_name_G1 = root + nam_G1_names[i]
        load_nam_cube_G1 = load_cube(nam_name_G1)
        this_nam_cube_G1 = numpy.float32(load_nam_cube_G1)

        nam_name_G2 = root + nam_G2_names[i]
        load_nam_cube_G2 = load_cube(nam_name_G2)
        this_nam_cube_G2 = numpy.float32(load_nam_cube_G2)

        nam_name_G3 = root + nam_G3_names[i]
        load_nam_cube_G3 = load_cube(nam_name_G3)
        this_nam_cube_G3 = numpy.float32(load_nam_cube_G3)

        #print('the shape should change and shuffle: ', this_nam_cube_G3.shape)




        nam_name_G4 = root + nam_G4_names[i]
        load_nam_cube_G4 = load_cube(nam_name_G4)
        this_nam_cube_G4= numpy.float32(load_nam_cube_G4)


        mixed_name = root + mixed_file_names[i]
        load_mixed_cube = load_cube(mixed_name)
        #print('The size of mixed training input cube with float64 format: ', getsizeof(mixed_cube))
        this_mixed_cube = numpy.float32(load_mixed_cube)
        #print('The size of mixed training input cube with float32 format: ', getsizeof(mixed_cube))
        #print()
        #print(mixed_cube.shape)

        sst_name = root + mur_file_names[i]
        load_sst_cube = load_cube(sst_name)
        #print('The size of mur training input cube with float64 format: ', getsizeof(mur_cube))
        this_mur_cube = numpy.float32(load_sst_cube)
        #print('The size of mur training input cube with float32 format: ', getsizeof(mur_cube))
        #print()
        #print(mur_cube.shape)


        if NAM_CUBE_G1 is None:
            NAM_CUBE_G1 = this_nam_cube_G1
        else:
            NAM_CUBE_G1 = numpy.concatenate((NAM_CUBE_G1, this_nam_cube_G1), axis = 0)
        #print('Training NAM input size: ', training_nam_cube.shape)
        del load_nam_cube_G1, this_nam_cube_G1
        #==============================================================================
        if NAM_CUBE_G2 is None:
            NAM_CUBE_G2 = this_nam_cube_G2
        else:
            NAM_CUBE_G2 = numpy.concatenate((NAM_CUBE_G2, this_nam_cube_G2), axis = 0)
        #print('Training NAM input size: ', training_nam_cube.shape)
        del load_nam_cube_G2, this_nam_cube_G2
        #==============================================================================
        if NAM_CUBE_G3 is None:
            NAM_CUBE_G3 = this_nam_cube_G3
        else:
            NAM_CUBE_G3 = numpy.concatenate((NAM_CUBE_G3, this_nam_cube_G3), axis = 0)
        #print('Training NAM input size: ', training_nam_cube.shape)
        del load_nam_cube_G3, this_nam_cube_G3
        #==============================================================================
        if NAM_CUBE_G4 is None:
            NAM_CUBE_G4 = this_nam_cube_G4
        else:
            NAM_CUBE_G4 = numpy.concatenate((NAM_CUBE_G4, this_nam_cube_G4), axis = 0)
        #print('Training NAM input size: ', training_nam_cube.shape)
        del load_nam_cube_G4, this_nam_cube_G4
        #==============================================================================
        if MIXED_CUBE is None:
            MIXED_CUBE = this_mixed_cube
        else:
            MIXED_CUBE = numpy.concatenate((MIXED_CUBE, this_mixed_cube), axis = 0)

        #print('Training MIXED input size: ', MIXED_CUBE.shape)
        del this_mixed_cube, load_mixed_cube
        #==============================================================================
        if SST_CUBE is None:
            SST_CUBE = this_mur_cube
        else:
            SST_CUBE = numpy.concatenate((SST_CUBE, this_mur_cube), axis = 0)
        #print('Training MUR input size: ', SST_CUBE.shape)
        del this_mur_cube, load_sst_cube

        #print('=======================================================================')
        #3print()

    return NAM_CUBE_G1, NAM_CUBE_G2, NAM_CUBE_G3, NAM_CUBE_G4, MIXED_CUBE, SST_CUBE


def load_Cat_cube_data(nam_G1_names, nam_G2_names, nam_G3_names,nam_G4_names, mixed_file_names, mur_file_names, file_names_root, years):
    X_Nam_G1, X_Nam_G2, X_Nam_G3, X_Nam_G4, X_mixed, X_mur = concat_cat_npz_cube_files(
        nam_G1_names, nam_G2_names, nam_G3_names,nam_G4_names, mixed_file_names, mur_file_names, file_names_root, years)


    X_Nam_G1 = numpy.expand_dims(X_Nam_G1, axis = -1)
    X_Nam_G2 = numpy.expand_dims(X_Nam_G2, axis = -1)
    X_Nam_G3 = numpy.expand_dims(X_Nam_G3, axis = -1)
    X_Nam_G4 = numpy.expand_dims(X_Nam_G4, axis = -1)
    X_mixed = numpy.expand_dims(X_mixed, axis = -1)
    X_mur = numpy.expand_dims(X_mur, axis = -1)
    print('NAM_G1 input size: ', X_Nam_G1.shape)
    print('NAM_G2 input size: ', X_Nam_G2.shape)
    print('NAM_G3 input size: ', X_Nam_G3.shape)
    print('NAM_G4 input size: ', X_Nam_G4.shape)
    print('MIXED input size: ', X_mixed.shape)
    print('MUR input size: ', X_mur.shape)


    return [X_Nam_G1, X_Nam_G2, X_Nam_G3, X_Nam_G4, X_mixed, X_mur]





#===============================================================================
#============================ Target preparation ===============================
#===============================================================================

def plot_visibility_cases(data, year, margin):
    vis = data['VIS_Cat'].value_counts()
    nan = data['VIS_Cat'].isna().sum()
    values = vis.values
    fvalues = numpy.insert(values, 0, nan)
    names = ["VIS_nan", "VIS > 4mi", "1mi< VIS<= 4mi", "VIS =<1mi"]
    df = pandas.DataFrame(columns = ["VIS-Cat", "VIS_Count"])
    df["VIS-Cat"] = names
    df["VIS_Count"] = fvalues
    # plot the count
    fig, ax = plt.subplots(figsize = FIG_DEFULT_SIZE)
    ax = sns.barplot(x = "VIS-Cat", y="VIS_Count", data=df,
                     palette="Blues_d")
    xlocs, xlabs = plt.xticks()
    plt.xlabel('Visibility class')
    plt.ylabel('The number of cases')
    txt = ('The number of visibility cases for {0}').format(year)
    plt.title(txt)
    for i, v in enumerate(df["VIS_Count"]):
        plt.text(xlocs[i] , v + margin, str(v),
                    fontsize=12, color='red',
                    horizontalalignment='center', verticalalignment='center')

    plt.show()



def reading_csv_target_file(csv_file_name):
    data = pandas.read_csv(csv_file_name, header=0, sep=',')

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

    return data

class targets():
    def __init__(self, targets_file_names, training_years, validation_years, testing_years, DEFAULT_TARGET_DIR_NAME, priority_calss):
        self.targets_file_names = targets_file_names
        self.DEFAULT_TARGET_DIR_NAME = DEFAULT_TARGET_DIR_NAME
        self.training_years   = training_years
        self.validation_years = validation_years
        self.testing_years    = testing_years
        self.priority_calss   = priority_calss

    def multiclass_target(self):

        training_targets   = pandas.DataFrame()
        validation_targets = pandas.DataFrame()
        testing_targets    = pandas.DataFrame()

        TRAIN_FRAMES = []
        for i in self.training_years:
            year_name = self.targets_file_names[i]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            TRAIN_FRAMES.append(year_data)
        training_targets = pandas.concat(TRAIN_FRAMES)
        #print(training_targets.shape)
        categorical_training_targets   = to_categorical(training_targets)
        #print(categorical_training_targets.shape)


        VALID_FRAMES = []
        for j in self.validation_years:
            year_name = self.targets_file_names[j]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            VALID_FRAMES.append(year_data)
        validation_targets = pandas.concat(VALID_FRAMES)
        #print(validation_targets.shape)
        categorical_validation_targets = to_categorical(validation_targets)
        #print(categorical_validation_targets.shape)

        TEST_FRAMES = []
        for k in self.testing_years:
            year_name = self.targets_file_names[k]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            TEST_FRAMES.append(year_data)
        testing_targets = pandas.concat(TEST_FRAMES)
        #print(testing_targets.shape)
        categorical_testing_targets    = to_categorical(testing_targets)
        #print(categorical_testing_targets.shape)

        return [training_targets, categorical_training_targets,
        validation_targets, categorical_validation_targets,
        testing_targets, categorical_testing_targets]



    def binary_target(self):

        training_targets   = pandas.DataFrame()
        validation_targets = pandas.DataFrame()
        testing_targets    = pandas.DataFrame()

        TRAIN_FRAMES = []
        for i in self.training_years:
            year_name = self.targets_file_names[i]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            TRAIN_FRAMES.append(year_data)
        training_targets = pandas.concat(TRAIN_FRAMES)
        #print(training_targets.shape)

        VALID_FRAMES = []
        for j in self.validation_years:
            year_name = self.targets_file_names[j]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            VALID_FRAMES.append(year_data)
        validation_targets = pandas.concat(VALID_FRAMES)
        #print(validation_targets.shape)

        TEST_FRAMES = []
        for k in self.testing_years:
            year_name = self.targets_file_names[k]
            file_name = self.DEFAULT_TARGET_DIR_NAME + year_name
            year_data = pandas.read_csv(file_name, header=0, sep=',')
            year_data = year_data['VIS_Cat']
            TEST_FRAMES.append(year_data)
        testing_targets = pandas.concat(TEST_FRAMES)
        #print(testing_targets.head())
        #print(testing_targets.shape)


        training_binary_targets, validation_binary_targets, testing_binary_targets = target_converter(
            training_targets, validation_targets, testing_targets, self.priority_calss)
        #print(numpy.count_nonzero(training_binary_targets == 1, axis = 0))
        categorical_training_targets   = to_categorical(training_binary_targets)
        categorical_validation_targets = to_categorical(validation_binary_targets)
        categorical_testing_targets    = to_categorical(testing_binary_targets)


        return  [training_binary_targets, categorical_training_targets,
        validation_binary_targets, categorical_validation_targets,
        testing_binary_targets, categorical_testing_targets]


def target_converter(training_csv, validation_csv, testing_csv, priority_calss):
    training_csv_file    = binary_target_convert(training_csv, priority_calss)
    validation_csv_file  = binary_target_convert(validation_csv, priority_calss)
    testing_csv_file     = binary_target_convert(testing_csv, priority_calss)

    return training_csv_file, validation_csv_file, testing_csv_file



def binary_target_convert(data, priority_calss):

    if priority_calss == 0:
        data = data.replace(0, 0)
        data = data.replace(1, 1)
        data = data.replace(2, 1)
        data = data.replace(3, 1)
        data = data.replace(4, 1)

    elif priority_calss == 1:
        data = data.replace(0, 0)
        data = data.replace(1, 0)
        data = data.replace(2, 1)
        data = data.replace(3, 1)
        data = data.replace(4, 1)
    elif priority_calss == 2:
        data = data.replace(0, 0)
        data = data.replace(1, 0)
        data = data.replace(2, 0)
        data = data.replace(3, 1)
        data = data.replace(4, 1)
    elif priority_calss == 3:
        data = data.replace(0, 0)
        data = data.replace(1, 0)
        data = data.replace(2, 0)
        data = data.replace(3, 0)
        data = data.replace(4, 1)
    elif priority_calss == 4:
        data = data.replace(0, 1)
        data = data.replace(2, 1)
        data = data.replace(3, 1)
        data = data.replace(4, 0)


    return data


#===============================================================================================================#
#================================================= Saliency Map ================================================#
#===============================================================================================================#
def run_gradcam(model_object, list_of_input_matrices, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    T = number of input tensors to the model

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        only one example (storm object).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :param target_class: Activation maps will be created for this class.  Must
        be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.
    :return: class_activation_matrix: Class-activation matrix.  Dimensions of
        this numpy array will be the spatial dimensions of whichever input
        tensor feeds into the target layer.  For example, if the given input
        tensor is 2-dimensional with M rows and N columns, this array will be
        M x N.
    """

    for q in range(len(list_of_input_matrices)):
        if list_of_input_matrices[q].shape[0] != 1:
            list_of_input_matrices[q] = numpy.expand_dims(
                list_of_input_matrices[q], axis=0)

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        if target_class == 1:
            # loss_tensor = model_object.layers[-1].output[..., 0]
            loss_tensor = model_object.layers[-1].input[..., 0]
        else:
            # loss_tensor = -1 * model_object.layers[-1].output[..., 0]
            loss_tensor = -1 * model_object.layers[-1].input[..., 0]
    else:
        # loss_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = model_object.layers[-1].input[..., target_class]

    # Create gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = utils._compute_gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = utils._normalize_tensor(gradient_tensor)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    gradient_function = K.function(
        list_of_input_tensors, [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        list_of_input_matrices)
    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation matrix.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.ones(
        target_layer_activation_matrix.shape[:-1])

    num_filters = mean_weight_by_filter.shape[1]
    for m in range(num_filters):
        class_activation_matrix = mean_weight_by_filter[:,m] * target_layer_activation_matrix[:,:,:, m]
        class_activation_matrix += class_activation_matrix

    spatial_dimensions = numpy.array(
        list_of_input_matrices[0].shape[1:-1], dtype=int)
    class_activation_matrix = utils._upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=spatial_dimensions)

    class_activation_matrix[class_activation_matrix < 0.] = 0.
    # denominator = numpy.maximum(numpy.max(class_activation_matrix), K.epsilon())
    # return class_activation_matrix / denominator

    return class_activation_matrix



def _do_saliency_calculations(
        cnn_model_object, loss_tensor, list_of_input_matrices):
    """Does the nitty-gritty part of computing saliency maps.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: list_of_saliency_matrices: length-T list of numpy arrays,
        comprising the saliency map for each example.
        list_of_saliency_matrices[i] has the same dimensions as
        list_of_input_matrices[i] and defines the "saliency" of each value x,
        which is the gradient of the loss function with respect to x.
    """

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.std(list_of_gradient_tensors[i]), K.epsilon()
        )

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        list_of_gradient_tensors
    )

    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0]
    )

    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices


def get_saliency_for_class(cnn_model_object, target_class,
                           list_of_input_matrices):
    """For each input example, creates saliency map for prob of given class.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Saliency maps will be created for probability of this
        class.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :return: list_of_saliency_matrices: Same.
    """

    target_class = int(numpy.round(target_class))
    assert target_class >= 0

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _do_saliency_calculations(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def plot_saliency_2d(
        saliency_matrix, axes_object, colour_map_object,
        max_absolute_contour_level, contour_interval, line_width=2):
    """Plots saliency map over 2-D grid (for one predictor).

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_absolute_contour_level: Max saliency to plot.  The minimum
        saliency plotted will be `-1 * max_absolute_contour_level`.
    :param max_absolute_contour_level: Max absolute saliency value to plot.  The
        min and max values, respectively, will be
        `-1 * max_absolute_contour_level` and `max_absolute_contour_level`.
    :param contour_interval: Saliency interval between successive contours.
    :param line_width: Width of contour lines.
    """

    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_level / contour_interval
    ))

    # Plot positive values.
    these_contour_levels = numpy.linspace(
        0., max_absolute_contour_level, num=half_num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='solid', zorder=1e6)

    # Plot negative values.
    these_contour_levels = these_contour_levels[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='dashed', zorder=1e6)


def plot_many_saliency_maps(
        saliency_matrix, axes_objects_2d_list, colour_map_object,
        max_absolute_contour_level, contour_interval, line_width=2):
    """Plots 2-D saliency map for each predictor.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param axes_objects_2d_list: See doc for `_init_figure_panels`.
    :param colour_map_object: See doc for `plot_saliency_2d`.
    :param max_absolute_contour_level: Same.
    :param max_absolute_contour_level: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    """

    num_predictors = saliency_matrix.shape[-1]
    num_panel_rows = len(axes_objects_2d_list)
    num_panel_columns = len(axes_objects_2d_list[0])

    for m in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            m, (num_panel_rows, num_panel_columns)
        )

        plot_saliency_2d(
            saliency_matrix=saliency_matrix[..., m],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval, line_width=line_width)


def _compute_gradients(loss_tensor, list_of_input_tensors):
    """Computes gradient of each input tensor with respect to loss tensor.

    :param loss_tensor: Loss tensor.
    :param list_of_input_tensors: 1-D list of input tensors.
    :return: list_of_gradient_tensors: 1-D list of gradient tensors.
    """

    list_of_gradient_tensors = tensorflow.gradients(
        loss_tensor, list_of_input_tensors)

    for i in range(len(list_of_gradient_tensors)):
        if list_of_gradient_tensors[i] is not None:
            continue

        list_of_gradient_tensors[i] = tensorflow.zeros_like(
            list_of_input_tensors[i])

    return list_of_gradient_tensors


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation matrix (CAM).

    CAM may be 1-D, 2-D, or 3-D.

    :param class_activation_matrix: numpy array containing 1-D, 2-D, or 3-D
        class-activation matrix.
    :param new_dimensions: numpy array of new dimensions.  If matrix is
        {1D, 2D, 3D}, this must be a length-{1, 2, 3} array, respectively.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float)
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float)

    if len(new_dimensions) == 1:
        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=1, s=0
        )

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float)
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1],
        dtype=float
    )

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=1, ky=1, s=0)

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float)
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2],
        dtype=float)

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear'
    )

    row_index_matrix, column_index_matrix, height_index_matrix = (
        numpy.meshgrid(row_indices_new, column_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1
    )

    return interp_object(query_point_matrix)


def run_gradcam1(model_object, list_of_input_matrices, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    T = number of input tensors to the model

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        only one example (storm object).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :param target_class: Activation maps will be created for this class.  Must
        be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.
    :return: class_activation_matrix: Class-activation matrix.  Dimensions of
        this numpy array will be the spatial dimensions of whichever input
        tensor feeds into the target layer.  For example, if the given input
        tensor is 2-dimensional with M rows and N columns, this array will be
        M x N.
    """

    for q in range(len(list_of_input_matrices)):
        if list_of_input_matrices[q].shape[0] != 1:
            list_of_input_matrices[q] = numpy.expand_dims(
                list_of_input_matrices[q], axis=0)

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        if target_class == 1:
            # loss_tensor = model_object.layers[-1].output[..., 0]
            loss_tensor = model_object.layers[-1].input[..., 0]
        else:
            # loss_tensor = -1 * model_object.layers[-1].output[..., 0]
            loss_tensor = -1 * model_object.layers[-1].input[..., 0]
    else:
        # loss_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = model_object.layers[-1].input[..., target_class]

    # Create gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = _compute_gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    gradient_function = K.function(
        list_of_input_tensors, [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        list_of_input_matrices)
    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation matrix.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.ones(
        target_layer_activation_matrix.shape[:-1])

    num_filters = mean_weight_by_filter.shape[1]
    for m in range(num_filters):
        class_activation_matrix = mean_weight_by_filter[:,m] * target_layer_activation_matrix[:,:,:, m]
        class_activation_matrix += class_activation_matrix

    spatial_dimensions = numpy.array(
        list_of_input_matrices[0].shape[1:-1], dtype=int)
    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=spatial_dimensions)

    class_activation_matrix[class_activation_matrix < 0.] = 0.
    # denominator = numpy.maximum(numpy.max(class_activation_matrix), K.epsilon())
    # return class_activation_matrix / denominator

    return class_activation_matrix


def _gradient_descent_for_bwo(
        cnn_model_object, loss_tensor, init_function_or_matrices,
        num_iterations, learning_rate):
    """Does gradient descent (the nitty-gritty part) for backwards optimization.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor, defining the loss function to be
        minimized.
    :param init_function_or_matrices: Either a function or list of numpy arrays.

    If function, will be used to initialize input matrices.  See
    `create_gaussian_initializer` for an example.

    If list of numpy arrays, these are the input matrices themselves.  Matrices
    should be processed in the exact same way that training data were processed
    (e.g., normalization method).  Matrices must also be in the same order as
    training matrices, and the [q]th matrix in this list must have the same
    shape as the [q]th training matrix.

    :param num_iterations: Number of gradient-descent iterations (number of
        times that the input matrices are adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of the loss function with respect to x.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_input_matrices` will have
        the exact same shape, just with different values.
    """

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    num_input_tensors = len(list_of_input_tensors)
    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors)
    )

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_input_matrices = copy.deepcopy(
            init_function_or_matrices)
    else:
        list_of_optimized_input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = numpy.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int
            )

            list_of_optimized_input_matrices[i] = init_function_or_matrices(
                these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0]
        )

        if numpy.mod(j, 100) == 0:
            print('Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0]
            ))

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate
            )

    print('Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0]
    ))

    return list_of_optimized_input_matrices


def bwo_for_class(
        cnn_model_object, target_class, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_BWO_ITERATIONS,
        learning_rate=DEFAULT_BWO_LEARNING_RATE):
    """Does backwards optimization to maximize probability of target class.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Synthetic input data will be created to maximize
        probability of this class.
    :param init_function_or_matrices: See doc for `_gradient_descent_for_bwo`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: Same.
    """

    target_class = int(numpy.round(target_class))
    num_iterations = int(numpy.round(num_iterations))

    assert target_class >= 0
    assert num_iterations > 0
    assert learning_rate > 0.
    assert learning_rate < 1.

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _gradient_descent_for_bwo(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)


def _create_smoothing_filter(
        smoothing_radius_px, num_half_filter_rows, num_half_filter_columns,
        num_channels):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_half_filter_rows: Number of rows in one half of filter.  Total
        number of rows will be 2 * `num_half_filter_rows` + 1.
    :param num_half_filter_columns: Same but for columns.
    :param num_channels: C in the above discussion.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """

    num_filter_rows = 2 * num_half_filter_rows + 1
    num_filter_columns = 2 * num_half_filter_columns + 1

    row_offsets_unique = numpy.linspace(
        -num_half_filter_rows, num_half_filter_rows, num=num_filter_rows,
        dtype=float)

    column_offsets_unique = numpy.linspace(
        -num_half_filter_columns, num_half_filter_columns,
        num=num_filter_columns, dtype=float)

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        column_offsets_unique, row_offsets_unique)

    pixel_offset_matrix = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    small_weight_matrix = numpy.exp(
        -pixel_offset_matrix ** 2 / (2 * smoothing_radius_px ** 2)
    )
    small_weight_matrix = small_weight_matrix / numpy.sum(small_weight_matrix)

    weight_matrix = numpy.zeros(
        (num_filter_rows, num_filter_columns, num_channels, num_channels)
    )

    for k in range(num_channels):
        weight_matrix[..., k, k] = small_weight_matrix

    return weight_matrix



def get_saliencymap_info(saliency_matrix):
    
    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]
    
    max_absolute_contour_level = numpy.percentile(
    numpy.absolute(saliency_matrix), 99)

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_level / (max_absolute_contour_level/10)
    ))

    # Plot positive values.
    these_contour_levels_p = numpy.linspace(
        0., max_absolute_contour_level, num=half_num_contours)
    
    these_contour_levels_n = these_contour_levels_p[1:]
    
    return x_coord_matrix, y_coord_matrix, these_contour_levels_p, these_contour_levels_n


def predictor_saliency_vis(pMatList, SMatrixs, Predictor_name, I_cmap, S_cmap): 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax3.tick_params(labelsize=12)
    ax4.tick_params(labelsize=12)
    fig.suptitle(Predictor_name, fontsize=16)

    
    
    min1 = numpy.min(pMatList[0])
    min2 = numpy.min(pMatList[1])
    min3 = numpy.min(pMatList[2])
    min4 = numpy.min(pMatList[3])
    
    max1 = numpy.min(pMatList[0])
    max2 = numpy.min(pMatList[1])
    max3 = numpy.min(pMatList[2])
    max4 = numpy.min(pMatList[3])
    
    vMin = min(min1, min2, min3, min4)
    vMax = max(max1, max2, max3, max4)
                                          

    
    ax1.pcolormesh(
        pMatList[0], cmap=I_cmap, norm=None,
        vmin=vMin, vmax=vMax, shading='flat',
        edgecolors='None')
    ax1.set_title('T1:initial', fontsize=14)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    
    x_coord_matrix1, y_coord_matrix1, these_contour_levels_p1, these_contour_levels_n1 = get_saliencymap_info(SMatrixs[0])
    
    # Plot positive values.
    ax1.contour(
        x_coord_matrix1, y_coord_matrix1, SMatrixs[0],
        these_contour_levels_p1, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_p1),
        vmax=numpy.max(these_contour_levels_p1), linewidths=2,
        linestyles='solid', zorder=1e4)
    # Plot negative values.
    ax1.contour(
        x_coord_matrix1, y_coord_matrix1, -SMatrixs[0],
        these_contour_levels_n1, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_n1),
        vmax=numpy.max(these_contour_levels_n1), linewidths=2,
        linestyles='dashed', zorder=1e4)
    
    
    
    #=========================================
    ax2.pcolormesh(
        pMatList[1], cmap=I_cmap, norm=None,
        vmin=vMin, vmax=vMax, shading='flat',
        edgecolors='None')
    ax2.set_title('T2:6hr', fontsize=14)
    #ax2.set_xticks([])
    ax2.set_yticks([])
    #
    
    x_coord_matrix2, y_coord_matrix2, these_contour_levels_p2, these_contour_levels_n2 = get_saliencymap_info(SMatrixs[1])
        # Plot positive values.
    ax2.contour(
        x_coord_matrix2, y_coord_matrix2, SMatrixs[1],
        these_contour_levels_p1, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_p2),
        vmax=numpy.max(these_contour_levels_p2), linewidths=2,
        linestyles='solid', zorder=1e4)
    # Plot negative values.
    ax2.contour(
        x_coord_matrix2, y_coord_matrix2, -SMatrixs[1],
        these_contour_levels_n1, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_n2),
        vmax=numpy.max(these_contour_levels_n2), linewidths=2,
        linestyles='dashed', zorder=1e4)
    #=========================================
    ax3.pcolormesh(
        pMatList[2], cmap=I_cmap, norm=None,
        vmin=vMin, vmax=vMax, shading='flat',
        edgecolors='None')
    ax3.set_title('T3:12hr', fontsize=14)
    #ax3.set_xticks([])
    ax3.set_yticks([])
    #
    x_coord_matrix3, y_coord_matrix3, these_contour_levels_p3, these_contour_levels_n3 = get_saliencymap_info(SMatrixs[2])
        # Plot positive values.
    ax3.contour(
        x_coord_matrix3, y_coord_matrix3, SMatrixs[2],
        these_contour_levels_p3, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_p3),
        vmax=numpy.max(these_contour_levels_p3), linewidths=2,
        linestyles='solid', zorder=1e4)
    # Plot negative values.
    ax3.contour(
        x_coord_matrix3, y_coord_matrix3, -SMatrixs[2],
        these_contour_levels_n3, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_n3),
        vmax=numpy.max(these_contour_levels_n3), linewidths=2,
        linestyles='dashed', zorder=1e4)
    #=========================================
    ax4.pcolormesh(
        pMatList[3], cmap=I_cmap, norm=None,
        vmin=vMin, vmax=vMax, shading='flat',
        edgecolors='None')
    ax4.set_title('T4:24hr', fontsize=14)
    #ax4.set_xticks([])
    ax4.set_yticks([])
    

    x_coord_matrix4, y_coord_matrix4, these_contour_levels_p4, these_contour_levels_n4 = get_saliencymap_info(SMatrixs[3])
    
    # Plot positive values.
    ax4.contour(
        x_coord_matrix4, y_coord_matrix4, SMatrixs[3],
        these_contour_levels_p4, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_p4),
        vmax=numpy.max(these_contour_levels_p4), linewidths=2,
        linestyles='solid', zorder=1e4)
    # Plot negative values.
    ax4.contour(
        x_coord_matrix4, y_coord_matrix4, -SMatrixs[3],
        these_contour_levels_n4, cmap=S_cmap,
        vmin=numpy.min(these_contour_levels_n4),
        vmax=numpy.max(these_contour_levels_n4), linewidths=2,
        linestyles='dashed', zorder=1e4)
    
    
    #plt.savefig('./figs/'+ Predictor_name + '_C0.png')
    plt.show()
    
def get_predicto_matrix_day(test_input_list, day):
    G1 = test_input_list[0][day, :, :, :, :]
    predictor_matrixs1 = numpy.expand_dims(G1, axis = 0)
    G2 = test_input_list[1][day, :, :, :, :]
    predictor_matrixs2 = numpy.expand_dims(G2, axis = 0)
    G3 = test_input_list[2][day, :, :, :, :]
    predictor_matrixs3 = numpy.expand_dims(G3, axis = 0)
    G4 = test_input_list[3][day, :, :, :, :]
    predictor_matrixs4 = numpy.expand_dims(G4, axis = 0)
    G5 = test_input_list[4][day, :, :, :, :]
    predictor_matrixs5 = numpy.expand_dims(G5, axis = 0)
    G6 = test_input_list[5][day, :, :, :, :]
    predictor_matrixs6 = numpy.expand_dims(G6, axis = 0)
    

    predictor_matrixs = [predictor_matrixs1,predictor_matrixs2, predictor_matrixs3, predictor_matrixs4, 
                         predictor_matrixs5, predictor_matrixs6]
    
    return predictor_matrixs




def plot_timep_predictor(pMatList, Predictor_name, _cmap): 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax3.tick_params(labelsize=12)
    ax4.tick_params(labelsize=12)
    fig.suptitle(Predictor_name, fontsize=16)

    #vmin = min(np.min(pMatList[0], np.min(pMatList[1], np.min(pMatList[2], np.min(pMatList[3]))
    #vmax = max(np.max(pMatList[0], np.max(pMatList[1], np.max(pMatList[2], np.max(pMatList[3]))
                                          
    vmin = -8
    vmax = 8
    
    ax1.pcolormesh(
        pMatList[0], cmap=_cmap, norm=None,
        vmin=vmin, vmax=vmax, shading='flat',
        edgecolors='None')
    ax1.set_title('T1:initial', fontsize=14)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    
    ax2.pcolormesh(
        pMatList[1], cmap=_cmap, norm=None,
        vmin=vmin, vmax=vmax, shading='flat',
        edgecolors='None')
    ax2.set_title('T2:6hr', fontsize=14)
    #ax2.set_xticks([])
    ax2.set_yticks([])
    #

    ax3.pcolormesh(
        pMatList[2], cmap=_cmap, norm=None,
        vmin=vmin, vmax=vmax, shading='flat',
        edgecolors='None')
    ax3.set_title('T3:12hr', fontsize=14)
    #ax3.set_xticks([])
    ax3.set_yticks([])
    #

    ax4.pcolormesh(
        pMatList[3], cmap=_cmap, norm=None,
        vmin=vmin, vmax=vmax, shading='flat',
        edgecolors='None')
    ax4.set_title('T4:24hr', fontsize=14)
    #ax4.set_xticks([])
    ax4.set_yticks([])
    

    plt.show()
    
def plot_saliency_predictor_time(predictors_list, saliency_list, day, group=None, predictor=None, cmap=None, plot_mode = None):
    
    if group == 'G1':
        Saliency = saliency_list[0]
        matrixs  = predictors_list[0]
        GNames = NETCDF_PREDICTOR_NAMES['Physical_G1']
        idx = GNames.index(predictor)
        PMat_T1 = matrixs[day, :, :, (idx*4), 0]
        PMat_T2 = matrixs[day, :, :, (idx*4)+1, 0]
        PMat_T3 = matrixs[day, :, :, (idx*4)+2, 0]
        PMat_T4 = matrixs[day, :, :, (idx*4)+3, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, (idx*4), 0]
        saliency_matrix2 = Saliency[0, :, :, (idx*4)+1, 0]
        saliency_matrix3 = Saliency[0, :, :, (idx*4)+2, 0]
        saliency_matrix4 = Saliency[0, :, :, (idx*4)+3, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        
    elif group == 'G2':
        Saliency = saliency_list[1]
        matrixs  = predictors_list[1]
        GNames = NETCDF_PREDICTOR_NAMES['Physical_G2']
        idx = GNames.index(predictor)
        PMat_T1 = matrixs[day, :, :, (idx*4), 0]
        PMat_T2 = matrixs[day, :, :, (idx*4)+1, 0]
        PMat_T3 = matrixs[day, :, :, (idx*4)+2, 0]
        PMat_T4 = matrixs[day, :, :, (idx*4)+3, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, (idx*4), 0]
        saliency_matrix2 = Saliency[0, :, :, (idx*4)+1, 0]
        saliency_matrix3 = Saliency[0, :, :, (idx*4)+2, 0]
        saliency_matrix4 = Saliency[0, :, :, (idx*4)+3, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        
    
    elif group == 'G3':
        Saliency = saliency_list[2]
        matrixs  = predictors_list[2]
        GNames = NETCDF_PREDICTOR_NAMES['Physical_G3']
        idx = GNames.index(predictor)
        PMat_T1 = matrixs[day, :, :, (idx*4), 0]
        PMat_T2 = matrixs[day, :, :, (idx*4)+1, 0]
        PMat_T3 = matrixs[day, :, :, (idx*4)+2, 0]
        PMat_T4 = matrixs[day, :, :, (idx*4)+3, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, (idx*4), 0]
        saliency_matrix2 = Saliency[0, :, :, (idx*4)+1, 0]
        saliency_matrix3 = Saliency[0, :, :, (idx*4)+2, 0]
        saliency_matrix4 = Saliency[0, :, :, (idx*4)+3, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        
    elif group == 'G4':
        Saliency = saliency_list[3]
        matrixs  = predictors_list[3]
        GNames = NETCDF_PREDICTOR_NAMES['Physical_G4']
        idx = GNames.index(predictor)
        PMat_T1 = matrixs[day, :, :, (idx*4), 0]
        PMat_T2 = matrixs[day, :, :, (idx*4)+1, 0]
        PMat_T3 = matrixs[day, :, :, (idx*4)+2, 0]
        PMat_T4 = matrixs[day, :, :, (idx*4)+3, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, (idx*4), 0]
        saliency_matrix2 = Saliency[0, :, :, (idx*4)+1, 0]
        saliency_matrix3 = Saliency[0, :, :, (idx*4)+2, 0]
        saliency_matrix4 = Saliency[0, :, :, (idx*4)+3, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        
    elif group == 'Mixed':
        Saliency = saliency_list[4]
        matrixs  = predictors_list[4]
        GNames = NETCDF_PREDICTOR_NAMES['Mixed']
        idx = GNames.index(predictor)
        
        PMat_T1 = matrixs[day, :, :, (idx*4), 0]
        PMat_T2 = matrixs[day, :, :, (idx*4)+1, 0]
        PMat_T3 = matrixs[day, :, :, (idx*4)+2, 0]
        PMat_T4 = matrixs[day, :, :, (idx*4)+3, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, (idx*4), 0]
        saliency_matrix2 = Saliency[0, :, :, (idx*4)+1, 0]
        saliency_matrix3 = Saliency[0, :, :, (idx*4)+2, 0]
        saliency_matrix4 = Saliency[0, :, :, (idx*4)+3, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        
    elif group == 'SST':
        Saliency = saliency_list[5]
        matrixs  = predictors_list[5]
        GNames = NETCDF_PREDICTOR_NAMES['SST']
        #idx = GNames.index(predictor)
        
        PMat_T1 = matrixs[day, :, :, 0, 0]
        PMat_T2 = matrixs[day, :, :, 0, 0]
        PMat_T3 = matrixs[day, :, :, 0, 0]
        PMat_T4 = matrixs[day, :, :, 0, 0]
        pMatList = [PMat_T1, PMat_T2, PMat_T3, PMat_T4]

        saliency_matrix1 = Saliency[0, :, :, 0, 0]
        saliency_matrix2 = Saliency[0, :, :, 0, 0]
        saliency_matrix3 = Saliency[0, :, :, 0, 0]
        saliency_matrix4 = Saliency[0, :, :, 0, 0]
        sMatList = [saliency_matrix1, saliency_matrix2, saliency_matrix3, saliency_matrix4]
        

    
    if plot_mode == 'C': 
        _  = predictor_saliency_vis(pMatList, sMatList, predictor, cmap, plt.cm.RdBu)
    else: 
        _  = plot_timep_predictor(sMatList, predictor, cmap_)
    
    
    

'''def test_eval(y, ypred, th = None): 
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
    return [POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]'''