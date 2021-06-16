import numpy
import matplotlib.pyplot as pyplot
import glob
import os.path
import json
import netCDF4
import seaborn as sns
import numpy
import pandas 
import scipy
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
import scipy.io as sio 
from scipy.io import loadmat
from numpy import load 
from numpy import savez_compressed 
from sys import getsizeof

import utils

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