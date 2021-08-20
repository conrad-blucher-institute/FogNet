import copy
import time
import calendar
import netCDF4
import numpy
from numpy import savez_compressed
from numpy import load
import pandas
import tensorflow
from tensorflow.keras.utils import to_categorical
import scipy.ndimage
import matplotlib.pyplot as pyplot
import seaborn as sns
import os.path
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

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


NETCDF_PREDICTOR_NAMES = {
    'OldOrder': [NETCDF_TMP_2m, NETCDF_TMP_975mb, NETCDF_TMP_950mb, NETCDF_TMP_925mb,
    NETCDF_TMP_900mb, NETCDF_TMP_875mb, NETCDF_TMP_850mb, NETCDF_TMP_825mb, NETCDF_TMP_800mb, NETCDF_TMP_775mb, NETCDF_TMP_750mb,
    NETCDF_TMP_725mb, NETCDF_TMP_700mb, NETCDF_UGRD_10m, NETCDF_VGRD_10m, NETCDF_FRICV, NETCDF_TKE_975mb,
    NETCDF_TKE_950mb, NETCDF_TKE_925mb, NETCDF_TKE_900mb, NETCDF_TKE_875mb, NETCDF_TKE_850mb, NETCDF_TKE_825mb, NETCDF_TKE_800mb,
    NETCDF_TKE_775mb, NETCDF_TKE_750mb, NETCDF_TKE_725mb, NETCDF_TKE_700mb, NETCDF_UGRD_975mb, NETCDF_UGRD_950mb,
    NETCDF_UGRD_925mb, NETCDF_UGRD_900mb, NETCDF_UGRD_875mb, NETCDF_UGRD_850mb, NETCDF_UGRD_825mb, NETCDF_UGRD_800mb, NETCDF_UGRD_775mb,
    NETCDF_UGRD_750mb, NETCDF_UGRD_725mb, NETCDF_UGRD_700mb, NETCDF_VGRD_975mb, NETCDF_VGRD_950mb, NETCDF_VGRD_925mb,
    NETCDF_VGRD_900mb, NETCDF_VGRD_875mb, NETCDF_VGRD_850mb, NETCDF_VGRD_825mb, NETCDF_VGRD_800mb, NETCDF_VGRD_775mb, NETCDF_VGRD_750mb,
    NETCDF_VGRD_725mb, NETCDF_VGRD_700mb, NETCDF_Q975, NETCDF_Q950, NETCDF_Q925, NETCDF_Q900, NETCDF_Q875, NETCDF_Q850, NETCDF_Q825, NETCDF_Q800,
    NETCDF_Q775,NETCDF_Q750, NETCDF_Q725, NETCDF_Q700,
    NETCDF_RH_975mb, NETCDF_RH_950mb, NETCDF_RH_925mb,NETCDF_RH_900mb, NETCDF_RH_875mb, NETCDF_RH_850mb, NETCDF_RH_825mb, NETCDF_RH_800mb,
    NETCDF_RH_775mb, NETCDF_RH_750mb, NETCDF_RH_725mb, NETCDF_RH_700mb, NETCDF_DPT_2m, NETCDF_Q, NETCDF_RH_2m, NETCDF_LCLT, NETCDF_VIS,
    NETCDF_VVEL_975mb, NETCDF_VVEL_950mb, NETCDF_VVEL_925mb, NETCDF_VVEL_900mb, NETCDF_VVEL_875mb, NETCDF_VVEL_850mb, NETCDF_VVEL_825mb,
    NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb],

    'NewOrder': [NETCDF_TMP_2m, NETCDF_TMP_975mb, NETCDF_TMP_950mb, NETCDF_TMP_925mb,
    NETCDF_TMP_900mb, NETCDF_TMP_875mb, NETCDF_TMP_850mb, NETCDF_TMP_825mb, NETCDF_TMP_800mb, NETCDF_TMP_775mb, NETCDF_TMP_750mb,
    NETCDF_TMP_725mb, NETCDF_TMP_700mb, NETCDF_UGRD_10m, NETCDF_VGRD_10m, NETCDF_FRICV, NETCDF_UGRD_975mb, NETCDF_VGRD_975mb, NETCDF_TKE_975mb,
    NETCDF_UGRD_950mb, NETCDF_VGRD_950mb, NETCDF_TKE_950mb, NETCDF_UGRD_925mb, NETCDF_VGRD_925mb, NETCDF_TKE_925mb, NETCDF_UGRD_900mb, NETCDF_VGRD_900mb,
    NETCDF_TKE_900mb, NETCDF_UGRD_875mb, NETCDF_VGRD_875mb, NETCDF_TKE_875mb, NETCDF_UGRD_850mb, NETCDF_VGRD_850mb, NETCDF_TKE_850mb, NETCDF_UGRD_825mb,
    NETCDF_VGRD_825mb, NETCDF_TKE_825mb, NETCDF_UGRD_800mb, NETCDF_VGRD_800mb, NETCDF_TKE_800mb, NETCDF_UGRD_775mb, NETCDF_VGRD_775mb,
    NETCDF_TKE_775mb, NETCDF_UGRD_750mb, NETCDF_VGRD_750mb, NETCDF_TKE_750mb, NETCDF_UGRD_725mb, NETCDF_VGRD_725mb, NETCDF_TKE_725mb,
    NETCDF_UGRD_700mb,  NETCDF_VGRD_700mb, NETCDF_TKE_700mb, NETCDF_Q975, NETCDF_Q950, NETCDF_Q925, NETCDF_Q900, NETCDF_Q875, NETCDF_Q850,
    NETCDF_Q825, NETCDF_Q800, NETCDF_Q775,NETCDF_Q750, NETCDF_Q725, NETCDF_Q700,
    NETCDF_RH_975mb, NETCDF_RH_950mb, NETCDF_RH_925mb,NETCDF_RH_900mb, NETCDF_RH_875mb, NETCDF_RH_850mb, NETCDF_RH_825mb, NETCDF_RH_800mb,
    NETCDF_RH_775mb, NETCDF_RH_750mb, NETCDF_RH_725mb, NETCDF_RH_700mb, NETCDF_DPT_2m, NETCDF_Q, NETCDF_RH_2m, NETCDF_LCLT, NETCDF_VIS,
    NETCDF_VVEL_975mb, NETCDF_VVEL_950mb, NETCDF_VVEL_925mb, NETCDF_VVEL_900mb, NETCDF_VVEL_875mb, NETCDF_VVEL_850mb, NETCDF_VVEL_825mb,
    NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb],

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
    NETCDF_VVEL_800mb, NETCDF_VVEL_775mb, NETCDF_VVEL_750mb, NETCDF_VVEL_725mb, NETCDF_VVEL_700mb]
    }


NETCDF_TMPDPT = 'TMP-DPT'
NETCDF_TMPSST = 'TMP-SST'
NETCDF_DPTSST = 'DPT-SST'
NETCDF_MUR_NAMES = [NETCDF_SST]
NETCDF_TMP_NAMES = [NETCDF_TMP_SFC, NETCDF_TMP_2m, NETCDF_DPT_2m]
NETCDF_MIXED_NAMES = [NETCDF_SST, NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST]
NETCDF_GEN_NAMES = [NETCDF_TMPDPT, NETCDF_TMPSST, NETCDF_DPTSST]


# Directories.
YEAR_FOG_DIR_NAME = '.'
ALL_FOG_DIR_NAME = '..'
DEFAULT_IMAGE_DIR_NAME = ('/data1/fog-data/fog-maps/')

#6HOURS
DEFAULT_TARGET_DIR_NAME = ('../Dataset/TARGET/')
SAVE_CUBE_DIR = '../Dataset/INPUT/MinMax/HIGH/'
SAVE_FILE_NAMES_DIR = '../Dataset/NAMES/'
SAVE_TARGET_DIR = '../Dataset/TARGET/'
DEFAULT_CUBES_12_DIR_NAME = ('../Dataset/INPUT/12Hours/')
DEFAULT_TARGET_DIR_NAME = ('../Dataset/TARGET/12Hours/')
#12HOURS
DEFAULT_12HOURS_TARGET_DIR = ('../Dataset/12HOURS/TARGET/')
DEFAULT_12HOURS_CUBES_DIR  = ('../Dataset/12HOURS/INPUT/')
DEFAULT_12HOURS_NAMES_DIR  = ('../Dataset/12HOURS/NAMES/')
#24HOURS
DEFAULT_24HOURS_TARGET_DIR = ('../Dataset/24HOURS/TARGET/')
DEFAULT_24HOURS_CUBES_DIR  = ('../Dataset/24HOURS/INPUT/')
DEFAULT_24HOURS_NAMES_DIR  = ('../Dataset/24HOURS/NAMES/')


### Defult Names and Settings
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'
FIG_DEFULT_SIZE = (12, 10)
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
CUBE_NAMES_KEY = 'cube_name'
SST_MATRIX_KEY = 'sst_matrix'
SST_NAME_KEY = 'sst_name'

# Misc constants.
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
TIME_CYCLE_FORMAT = '[0-9][0-9][0][0]'
HOUR_PREDICTION_FORMAT = '[0-9][0-9][0-9]'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'


#==========================================================================================#
#================================ preprocessing tools =====================================#
#==========================================================================================#
def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.
    Unix format = seconds since 0000 UTC 1 Jan 1970.
    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """
    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.
    Unix format = seconds since 0000 UTC 1 Jan 1970.
    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """
    return time.strftime(time_format, time.gmtime(unix_time_sec))


def _nc_file_name_to_date(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.
    :param netcdf_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """
    pathless_file_name = os.path.split(netcdf_file_name)[-1]
    date_string = pathless_file_name.replace(pathless_file_name[0:5], '').replace(
        pathless_file_name[-18:], '')
    # Verify.
    time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string

def _nc_file_name_to_timecycle(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.
    :param netcdf_file_name: Path to input file.
    :return: time-cycle prediction.
    """
    pathless_file_name = os.path.split(netcdf_file_name)[-1]
    timecycle_string = pathless_file_name.replace(pathless_file_name[0:14], '').replace(
        pathless_file_name[-13:], '')
    # Verify.
    #time_string_to_unix(time_string=timecycle_string, time_format=TIME_CYCLE_FORMAT)
    return timecycle_string

def _nc_file_name_to_hourprediction(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.
    :param netcdf_file_name: Path to input file.
    :return: time-cycle prediction.
    """
    pathless_file_name = os.path.split(netcdf_file_name)[-1]
    hourpredic_string = pathless_file_name.replace(pathless_file_name[0:19], '').replace(
        pathless_file_name[-9:], '')
    return hourpredic_string

def find_match_name (netcdf_file_name):
    date = _nc_file_name_to_date(netcdf_file_name)
    timecycle = _nc_file_name_to_timecycle(netcdf_file_name)
    hourprediction = _nc_file_name_to_hourprediction(netcdf_file_name)
    this_cube_match_name = date + timecycle

    return {'cube_match_name' : this_cube_match_name,
    'date' : date,
    'timecycle' : timecycle,
    'hourprediction' : hourprediction}

def netcdf_names_check(root_dir, target=None):
    # Reading Target data and list all NAM index to check and remove them plus corresponding map data:
    nan_index = target[target['VIS_Cat'].isnull().values].index.tolist()
    Nannames = target['Date'].iloc[nan_index]
    names = Nannames.values
    NAN = pandas.DataFrame(columns = ['name'])
    NAN['name'] = names
    # Reading the directory of map data and check and remove those they are incomplete or target is NAN!
    netcef_nams_file_name = []   # we need to return the name of maps which are coomplete!
    netcef_murs_file_name = []   # we need to return the name of maps which are coomplete!
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        datavalume = len(files)
        if datavalume == 149:
            namesplit = os.path.split(files[0])[-1]
            match_name_1 = namesplit.replace(namesplit[:5], '').replace(namesplit[13:], '')
            for f in NAN['name'].isin([int(match_name_1)]):
                if f is True:
                    foldercondition = False
                    #print(('The Traget for "{0}" day is NAN!').format(match_name_1))
                    #print('Removed the corresponding map for days with NAN Target!')
                    #print('=====================================================================')
                    break
                else:
                    foldercondition = True
            if (foldercondition is True):
                for name in files:
                    namesplit = os.path.split(name)[-1]
                    namOrmur = namesplit.replace(namesplit[4:], '')
                    if (namOrmur == 'murs'):
                        name = root +'/'+ name
                        netcef_murs_file_name.append(name)
                        netcef_murs_file_name.sort()
                    elif (namOrmur == 'maps'):
                        name = root +'/'+ name
                        netcef_nams_file_name.append(name)
                        netcef_nams_file_name.sort()

        elif datavalume < 149 and datavalume != 0:
            if files[0].endswith(".txt"):
                print('break')
            else:
                namesplit = os.path.split(files[0])[-1]
                match_name = namesplit.replace(namesplit[:5], '').replace(namesplit[13:], '')
                #print(('The expected maps is 149 which there are "{0}" maps for {1} day!').format(datavalume, match_name))
                target = target.drop(target[target.Date == int(match_name)].index)
                #print('Removed the corresponding target values for days with incomplete data!')
                #print('=====================================================================')

    target = target.dropna()
    target = RenewDf(target)
    for d in target['Date']:
        if target.loc[target.Date == d, 'Date'].count() < 4:
            target = target.drop(target.loc[target.Date == d, 'Date'].index)
    target = RenewDf(target)
    return [netcef_nams_file_name, netcef_murs_file_name, target]


def RenewDf(df):
    newdf = pandas.DataFrame(columns=['Date', 'VIS', 'VIS_Cat'])
    dates = df['Date'].values
    cat = df['VIS_Cat'].values
    vis = df['VIS'].values
    newdf['Date'] = dates
    newdf['VIS_Cat'] = cat
    newdf['VIS'] = vis
    return newdf

def copy_mur_name_ntimes(netcdf_file_names, output, n):

    for i in range(len(netcdf_file_names)):
        name = netcdf_file_names[i]
        for j in range(n):
            output.append(name)

    return output

def map_upsampling(downsampled_cube_file):
    upsampled_map = None
    upsampled_map = scipy.ndimage.zoom(downsampled_cube_file, 11.75, order=3)
    return upsampled_map

def map_downsampling(upsampled_cube_file):
    downsampled_map = None
    downsampled_map = scipy.ndimage.zoom(upsampled_cube_file, 0.0851, order=3)
    return downsampled_map



#===========================================================================================#
#===============  Finding the cubes based on their names ===================================#
#===========================================================================================#
def find_map_name_date(first_date_string, last_date_string, target = None, image_dir_name = DEFAULT_IMAGE_DIR_NAME):
    """Finds image (NetCDF) files in the given date range.
    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :param image_dir_name: Name of directory with image (NetCDF) files.
    :return: netcdf_file_names: 1-D list of paths to image files.
    """
# check the target and return the desierd target index:
    Dates = target['Date'].values
    good_indices_target = numpy.where(numpy.logical_and(
        Dates >= int(first_date_string),
        Dates <= int(last_date_string)
    ))[0]
    input_target = target.take(good_indices_target)
    target_1 = RenewDf(input_target)

    netcdf_nam_file_names, netcdf_mur_file_names, target_2 = netcdf_names_check(image_dir_name, target_1)
    target = RenewDf(target_2)



    first_time_unix_sec = time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT)
    last_time_unix_sec = time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT)

    # NAM Data
    file_date_strings = [_nc_file_name_to_date(f) for f in netcdf_nam_file_names]
    file_times_unix_sec = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices_nam = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    # MUR Data
    file_date_strings_mur = [_nc_file_name_to_date(f) for f in netcdf_mur_file_names]
    file_times_unix_sec_mur = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings_mur
    ], dtype=int)

    good_indices_mur = numpy.where(numpy.logical_and(
        file_times_unix_sec_mur >= first_time_unix_sec,
        file_times_unix_sec_mur <= last_time_unix_sec
    ))[0]



    return [netcdf_nam_file_names[k] for k in good_indices_nam], [netcdf_mur_file_names[k] for k in good_indices_mur], target


def find_nam_cubes_name_hourpredict(netcdf_file_names, hour_prediction_names = ['000', '006', '012', '024']):
    """Depend on the time prediction this function just select the name of selected maps:
    for example in this case, the time prediction 000, 003 and 006 hour has been selected.
    """
    file_date_strings = [_nc_file_name_to_hourprediction(f) for f in netcdf_file_names]
    file_date_strings = pandas.DataFrame(file_date_strings, columns = ['str'])
    good_indices = file_date_strings[
        (file_date_strings['str'] == hour_prediction_names[0]) |
        (file_date_strings['str'] == hour_prediction_names[1]) |
        (file_date_strings['str'] == hour_prediction_names[2]) |
        (file_date_strings['str'] == hour_prediction_names[3])]
    return [netcdf_file_names[k] for k in list(good_indices.index)]




#============================================================================
#====================      Reading Nam and MUR maps   =======================
#============================================================================
def read_nam_maps(netcdf_file_name, PREDICTOR_NAMES):
    """Reads fog-centered maps from NetCDF file.
    E = number of examples (fog objects) in file
    M = number of rows in each fog-centered grid
    N = number of columns in each fog-centered grid
    C = number of channels (predictor variables)
    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    """
    NETCDF_PREDICTOR_NAMES = PREDICTOR_NAMES
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    lons = numpy.array(dataset_object.variables[NETCDF_LONGITUDE][:], dtype=float)
    lats = numpy.array(dataset_object.variables[NETCDF_LATITUDE][:], dtype=float)

    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float
        )

        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        PREDICTOR_NAMES_KEY: NETCDF_PREDICTOR_NAMES,
        NETCDF_LONGITUDE: lons,
        NETCDF_LATITUDE: lats}


def read_mur_map(netcdf_file_name, PREDICTOR_NAMES):

    NETCDF_PREDICTOR_NAMES = PREDICTOR_NAMES
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    lons = numpy.array(dataset_object.variables[MUR_LONGITUDE][:], dtype=float)
    lats = numpy.array(dataset_object.variables[MUR_LATITUDE][:], dtype=float)

    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float
        )

        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        PREDICTOR_NAMES_KEY: NETCDF_PREDICTOR_NAMES,
        MUR_LONGITUDE: lons,
        MUR_LATITUDE: lats}

def read_many_nam_cube(netcdf_file_names, PREDICTOR_NAMES):
    """Reads storm-centered images from many NetCDF files.
    :param netcdf_file_names: 1-D list of paths to input files.
    :return: image_dict: See doc for `read_image_file`.
    """
    image_dict = None
    keys_to_concat = [PREDICTOR_MATRIX_KEY]

    for this_file_name in netcdf_file_names:
        #print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_nam_maps(this_file_name, PREDICTOR_NAMES)

        if image_dict is None:
            image_dict = copy.deepcopy(this_image_dict)
            continue
        for this_key in keys_to_concat:
            image_dict[this_key] = numpy.concatenate(
                (image_dict[this_key], this_image_dict[this_key]), axis=0
            )
    return image_dict


#======================================================================================================
#=========================== Concatenate the cubes in order to create Tensor: =========================
#======================================================================================================

# in this section there are two function just to generate three manually features include of
# [TMPsurface -SST, TMPsurface-DPT, DPT -SST]
def highres_features(nam_file_name, nam_feature_name, mur_file_name, mur_feature_name):
    """
    to generate the features first it needs to upsample the nam features to the same resolution of SST maps
    , then geerate the new feattures and make a cube form them.
    """

    keys_to_concat = [PREDICTOR_MATRIX_KEY]

    this_nam_file_name = nam_file_name

    this_nam_dict      = read_nam_maps(this_nam_file_name, nam_feature_name)
    #print(this_nam_dict[PREDICTOR_MATRIX_KEY].shape)
    this_nam_tmpsurf   = this_nam_dict[PREDICTOR_MATRIX_KEY][0, :,:,0]
    this_nam_tmp2m     = this_nam_dict[PREDICTOR_MATRIX_KEY][0, :,:,1]
    this_nam_dpt       = this_nam_dict[PREDICTOR_MATRIX_KEY][0, :,:,2]
    #print('BEFORE UPSAMPLING: ', this_nam_dpt.shape)

    up_this_nam_tmpsurf= map_upsampling(this_nam_tmpsurf)
    up_this_nam_tmp2m  = map_upsampling(this_nam_tmp2m)
    up_this_nam_dpt    = map_upsampling(this_nam_dpt)
    #print('AFTER UPSAMPLING: ', up_this_nam_dpt.shape)

    this_mur_file_name = mur_file_name
    this_mur_dict      = read_mur_map(this_mur_file_name, mur_feature_name)
    this_mur_file      = this_mur_dict[PREDICTOR_MATRIX_KEY][0, :, :, 0]
    #print('MUR SIZE: ', this_mur_file.shape)
    # filling the mur map with tmp surface:
    for l in range(len(this_mur_file)):
        for w in range(len(this_mur_file)):
            if this_mur_file[l, w] == -32768:
                this_mur_file[l, w] = up_this_nam_tmpsurf[l, w]


    NETCDF_SST         = this_mur_file
    NETCDF_TMPDPT      = numpy.subtract(up_this_nam_tmp2m , up_this_nam_dpt)
    NETCDF_TMPSST      = numpy.subtract(up_this_nam_tmp2m , this_mur_file)
    NETCDF_DPTSST      = numpy.subtract(up_this_nam_dpt , this_mur_file)

    # downsampling:

    NETCDF_TMPDPT      = map_downsampling(NETCDF_TMPDPT)
    NETCDF_TMPSST      = map_downsampling(NETCDF_TMPSST)
    NETCDF_DPTSST      = map_downsampling(NETCDF_DPTSST)


    THIS_NETCDF_SST    = numpy.expand_dims(NETCDF_SST, axis=-1)
    THIS_NETCDF_TMPDPT = numpy.expand_dims(NETCDF_TMPDPT, axis=-1)
    THIS_NETCDF_TMPSST = numpy.expand_dims(NETCDF_TMPSST, axis=-1)
    THIS_NETCDF_DPTSST = numpy.expand_dims(NETCDF_DPTSST, axis=-1)

    THIS_NETCDF_SST_CUBE = numpy.expand_dims(THIS_NETCDF_SST, axis=0)


    THIS_NETCDF_MIXED_CUBE = numpy.concatenate((THIS_NETCDF_TMPDPT, THIS_NETCDF_TMPSST, THIS_NETCDF_DPTSST),
                                               axis = -1)
    THIS_NETCDF_MIXED_CUBE = numpy.expand_dims(THIS_NETCDF_MIXED_CUBE, axis=0)

    return { SST_MATRIX_KEY:   THIS_NETCDF_SST_CUBE,
        PREDICTOR_MATRIX_KEY: THIS_NETCDF_MIXED_CUBE,
        PREDICTOR_NAMES_KEY: NETCDF_MIXED_NAMES}

def SST_Upsampling(lowe_res_sst_cube):

    length = lowe_res_sst_cube.shape[0]

    high_res_sst_cube = None

    for m in range(length):
        low_res_map = lowe_res_sst_cube[m, :, : , 0]
        this_high_res_map = scipy.ndimage.zoom(low_res_map, 1.021, order=3)
        this_high_res_map = numpy.expand_dims(this_high_res_map, axis = 0)

        if high_res_sst_cube is None:
            high_res_sst_cube = this_high_res_map
        else:
            high_res_sst_cube = numpy.concatenate([high_res_sst_cube, this_high_res_map], axis = 0)

    high_res_sst_cube = numpy.expand_dims(high_res_sst_cube, axis = -1)

    return high_res_sst_cube

def SST_384_cubes(sst_376_cube_names, cubes_dir):

    for key in sst_376_cube_names:
        print(('Process for {0} just started!').format(key))
        low_res_cube_name = sst_376_cube_names[key][0]
        low_res_cube_name = cubes_dir + low_res_cube_name
        low_res_cube      = load_cube(low_res_cube_name)
        high_res_cube     = SST_Upsampling(low_res_cube)
        print('The shape after upsampling: ', high_res_cube.shape)

        sst_cube_name = 'NETCDF_SST_CUBE_' + key + '.npz'
        sst_cube_path = os.path.join(SAVE_CUBE_DIR, sst_cube_name)
        savez_compressed(sst_cube_path, high_res_cube)







def mixed_cubes(nam_file_names, nam_feature_name, mur_file_names, mur_feature_name):

    length = len(nam_file_names)
    NETCDF_MIXED_CUBES = None
    NETCDF_SST_CUBES = None

    for i in range(length):
        this_nam_file_name = nam_file_names[i]
        this_mur_file_name = mur_file_names[i]
        this_highres_cube  = highres_features(this_nam_file_name, nam_feature_name, this_mur_file_name, mur_feature_name)


        if NETCDF_MIXED_CUBES is None:
            NETCDF_MIXED_CUBES = this_highres_cube[PREDICTOR_MATRIX_KEY]
            NETCDF_SST_CUBES   = this_highres_cube[SST_MATRIX_KEY]
        else:
            NETCDF_MIXED_CUBES = numpy.concatenate((NETCDF_MIXED_CUBES, this_highres_cube[PREDICTOR_MATRIX_KEY]), axis = 0)
            NETCDF_SST_CUBES = numpy.concatenate((NETCDF_SST_CUBES, this_highres_cube[SST_MATRIX_KEY]), axis = 0)
    return {SST_MATRIX_KEY : NETCDF_SST_CUBES,
        PREDICTOR_MATRIX_KEY: NETCDF_MIXED_CUBES,
        PREDICTOR_NAMES_KEY: NETCDF_GEN_NAMES,
        SST_NAME_KEY: NETCDF_MUR_NAMES
        }


#======================================================================================================
#=========================== Concatenate the cubes in order to create Tensor: =========================
#======================================================================================================
def concate_nam_cubes_files(netcdf_file_names, PREDICTOR_NAMES):
    """
    concatenate the input maps for each day based on lead time prediction.
    for instance the lead time is 6 hours and there are 3 cubes per each time cycle include 000, 003 and 006
    based on "find_nam_cubes_name_hourpredict" function their names are selected, and this function creates the cube
    of all three time prediction using concatenation.
    input: netcdf_file_names
    output: 3D cube
    """
    cube_tensor = None
    cubes_dict = {}
    cancat_cube = None
    match_name = None
    cubenumber = 0
    Depth = 4*len(PREDICTOR_NAMES)


    cube_names = netcdf_file_names[1]
    for this_file in range(len(cube_names)):
        this_cube_name = cube_names[this_file]
        this_cube_name_details = find_match_name (this_cube_name)
        this_cube_match_name = this_cube_name_details ['cube_match_name']
        this_cube_date_name = this_cube_name_details ['date']
        this_cube_timecycle_name = this_cube_name_details ['timecycle']
        #print('Name this_cube_match_name before if: ', this_cube_match_name)
        this_cube_tensor = netcdf_file_names[0][this_file]
        #print('Size this_cube_tensor before if: ', this_cube_tensor['predictor_matrix'].shape)

        if cube_tensor is None:
            cube_tensor = this_cube_tensor
            match_name = this_cube_match_name
            #print('Name cube_match_name after if: ', cube_match_name)
            #print('Size cube_tensor after if: ', cube_tensor['predictor_matrix'].shape)
        #print(this_cube_match_name)
        elif match_name == this_cube_match_name:
            #print(True)
            cube_tensor = numpy.concatenate((cube_tensor[PREDICTOR_MATRIX_KEY], this_cube_tensor[PREDICTOR_MATRIX_KEY]), axis=-1)
            cube_tensor = {PREDICTOR_MATRIX_KEY: cube_tensor}
            #print('New Size of cube: ', cube_tensor['predictor_matrix'].shape)

            Depth_cube = cube_tensor[PREDICTOR_MATRIX_KEY].shape[3]
            if Depth_cube == Depth:
                cube_name = 'cube_' + this_cube_date_name + '_' + this_cube_timecycle_name +'_036' + '_input.nc'
                cube_tensor = {PREDICTOR_MATRIX_KEY: cube_tensor[PREDICTOR_MATRIX_KEY],
                              PREDICTOR_NAMES_KEY: 4 * PREDICTOR_NAMES,
                              CUBE_NAMES_KEY: cube_name}

                cubes_dict[cubenumber] = cube_tensor
                cubenumber = cubenumber + 1

        else:
            match_name = this_cube_match_name
            #print('Name cube_match_name after else: ', cube_match_name)
            cube_tensor = this_cube_tensor
            #print('Size cube_tensor after else: ', cube_tensor['predictor_matrix'].shape)
    return cubes_dict



def nam_cubes_dict(netcdf_file_names):

    cubes_dict = None
    keys_to_concat = [PREDICTOR_MATRIX_KEY]

    numerator = len(netcdf_file_names)

    for this_file in range(numerator):

        cube_tensor = netcdf_file_names[this_file]
        #print('Size this_cube_tensor before if: ', this_cube_tensor['predictor_matrix'].shape
        if cubes_dict is None:
            cubes_dict = copy.deepcopy(cube_tensor)

        else:
            cubes_dict = numpy.concatenate(
                (cubes_dict[PREDICTOR_MATRIX_KEY], cube_tensor[PREDICTOR_MATRIX_KEY]), axis=0
                    )
            cubes_dict = {PREDICTOR_MATRIX_KEY: cubes_dict}
    return cubes_dict


def concate_mixed_cubes_files(normalized_mur_cubes):
    """
    Mixed_cube means three generated feature manually include TMPsurface-SST, DPT-SST and TMPsurface-DPT.

    This function using the name of maps generate the cube with depth 3 of having three mentioned maps.
    """

    cancat_mur_cube = None
    length = len(normalized_mur_cubes)

    i = 0
    while i < length:

        m00 = normalized_mur_cubes[i][PREDICTOR_MATRIX_KEY]
        m06 = normalized_mur_cubes[i+1][PREDICTOR_MATRIX_KEY]
        m09 = normalized_mur_cubes[i+2][PREDICTOR_MATRIX_KEY]
        m12 = normalized_mur_cubes[i+3][PREDICTOR_MATRIX_KEY]
        this_cube_timecycle = numpy.concatenate(
            (m00, m06, m09, m12), axis=-1)
        this_cube_timecycle = numpy.expand_dims(
            this_cube_timecycle, axis=0)

        if cancat_mur_cube is None:
            cancat_mur_cube = copy.deepcopy(this_cube_timecycle)

        else:
            cancat_mur_cube = numpy.concatenate(
                (cancat_mur_cube, this_cube_timecycle), axis=0
                    )
        i+=4

    return cancat_mur_cube

def concate_sst_cubes_files(normalized_mur_cubes):

    cancat_sst_cube = None
    length = len(normalized_mur_cubes)

    i = 0
    while i < length:

        m06 = normalized_mur_cubes[i+2][SST_MATRIX_KEY]
        this_cube_timecycle = numpy.expand_dims(
            m06, axis=0)

        if cancat_sst_cube is None:
            cancat_sst_cube = copy.deepcopy(this_cube_timecycle)

        else:
            cancat_sst_cube = numpy.concatenate(
                (cancat_sst_cube, this_cube_timecycle), axis=0
                    )
        i+=4
    return cancat_sst_cube

#============================================================================
#====================      Normalization Step   =============================
#============================================================================

def _update_normalization_params(intermediate_normalization_dict, new_values):
    """Updates normalization params for one predictor.

    :param intermediate_normalization_dict: Dictionary with the following keys.
    intermediate_normalization_dict['num_values']: Number of values on which
        current estimates are based.
    intermediate_normalization_dict['mean_value']: Current estimate for mean.
    intermediate_normalization_dict['mean_of_squares']: Current mean of squared
        values.

    :param new_values: numpy array of new values (will be used to update
        `intermediate_normalization_dict`).
    :return: intermediate_normalization_dict: Same as input but with updated
        values.
    """

    if MEAN_VALUE_KEY not in intermediate_normalization_dict:
        intermediate_normalization_dict = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_VALUE_KEY], numpy.mean(new_values)
    ])
    these_weights = numpy.array([
        intermediate_normalization_dict[NUM_VALUES_KEY], new_values.size
    ])

    intermediate_normalization_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights)

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_values ** 2)
    ])

    intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights)

    intermediate_normalization_dict[NUM_VALUES_KEY] += new_values.size
    return intermediate_normalization_dict


def _get_standard_deviation(intermediate_normalization_dict):
    """Computes stdev from intermediate normalization params.

    :param intermediate_normalization_dict: See doc for
        `_update_normalization_params`.
    :return: standard_deviation: Standard deviation.
    """

    num_values = float(intermediate_normalization_dict[NUM_VALUES_KEY])
    multiplier = num_values / (num_values - 1)

    return numpy.sqrt(multiplier * (
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] -
        intermediate_normalization_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_nam_normalization_params(netcdf_file_names, PREDICTOR_NAMES):
    """Computes normalization params (mean and stdev) for each predictor.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None

    for this_file_name in netcdf_file_names:
        #print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_nam_maps(this_file_name, PREDICTOR_NAMES)

        if predictor_names is None:
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)

        for m in range(len(predictor_names)):
            norm_dict_by_predictor[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor[m],
                new_values=this_image_dict[PREDICTOR_MATRIX_KEY][..., m]
            )

    print('\n')
    normalization_dict = {}

    for m in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[m])

        normalization_dict[predictor_names[m]] = numpy.array(
            [this_mean, this_stdev]
        )

        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(predictor_names[m], this_mean, this_stdev)
        print(message_string)


    return normalization_dict


def normalize_sst_map(
        predictor_matrix, predictor_names, normalization_dict=None):


    normalization_dict = {}
    this_mean = numpy.mean(predictor_matrix)
    this_stdev = numpy.std(predictor_matrix, ddof=1)

    normalization_dict = numpy.array(
        [this_mean, this_stdev]
    )

    predictor_matrix = (
            (predictor_matrix - this_mean) / float(this_stdev))

    return {
        SST_MATRIX_KEY: predictor_matrix,
        SST_NAME_KEY: predictor_names
    }


def get_sst_normalization_params(NETCDF_HIGHRES_CUBES, PREDICTOR_NAMES):
    """Computes normalization params (mean and stdev) for each predictor.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None
    length = NETCDF_HIGHRES_CUBES[SST_MATRIX_KEY].shape[0]
    #print(length)

    for i in range(length):
        #print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = NETCDF_HIGHRES_CUBES[SST_MATRIX_KEY][i, :, :, :]

        if predictor_names is None:
            predictor_names = NETCDF_HIGHRES_CUBES[SST_NAME_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)

        norm_dict_by_predictor = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor,
                new_values = this_image_dict
            )

    print('\n')
    normalization_dict = {}


    this_mean = norm_dict_by_predictor[MEAN_VALUE_KEY]
    this_stdev = _get_standard_deviation(norm_dict_by_predictor)

    normalization_dict = numpy.array([this_mean, this_stdev])

    message_string = (
        'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format('SST', this_mean, this_stdev)
    print(message_string)

    return normalization_dict


## this function work when we are using 4 high resolution maps cube!!!
def get_mixed_normalization_params(NETCDF_HIGHRES_CUBES, PREDICTOR_NAMES):

    predictor_names = None
    norm_dict_by_predictor = None
    length = NETCDF_HIGHRES_CUBES[PREDICTOR_MATRIX_KEY].shape[0]
    #print(length)

    for i in range(length):
        #print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = NETCDF_HIGHRES_CUBES[PREDICTOR_MATRIX_KEY][i, :, :, :]
        #print(this_image_dict.shape)

        if predictor_names is None:
            predictor_names = NETCDF_HIGHRES_CUBES[PREDICTOR_NAMES_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)
            #print(len(predictor_names))
        for m in range(len(predictor_names)):
            norm_dict_by_predictor[m] = _update_normalization_params(
                intermediate_normalization_dict = norm_dict_by_predictor[m],
                new_values=this_image_dict[..., m]
            )

    normalization_dict = {}

    for m in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[m])

        normalization_dict[predictor_names[m]] = numpy.array(
            [this_mean, this_stdev]
        )

        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(predictor_names[m], this_mean, this_stdev)
        print(message_string)


def normalize_nam_maps(
        predictor_matrix, predictor_names, normalization_dict=None):
    """Normalizes images to z-scores.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_matrix[..., m])
            this_stdev = numpy.std(predictor_matrix[..., m], ddof=1)

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev]
            )

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            (predictor_matrix[..., m] - this_mean) / float(this_stdev)
        )

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        PREDICTOR_NAMES_KEY: predictor_names
    }



def denormalize_nam_maps(predictor_matrix, predictor_names, normalization_dict):
    """Denormalizes images from z-scores back to original scales.

    :param predictor_matrix: See doc for `normalize_images`.
    :param predictor_names: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input.
    """

    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            this_mean + this_stdev * predictor_matrix[..., m]
        )

    return predictor_matrix




def normalize_many_cubes(netcdf_file_names, normalization_dict, predictor_names):

    normmalized_cubes_dict = {}
    for m in range(len(netcdf_file_names)):
        this_cube = read_nam_maps(netcdf_file_names[m], predictor_names)
        normmalized_cubes_dict[m] = normalize_nam_maps(
        predictor_matrix = this_cube[PREDICTOR_MATRIX_KEY],
        predictor_names = this_cube[PREDICTOR_NAMES_KEY],
        normalization_dict = normalization_dict)

    message_string = (
        'The normalization of ' + 'netcdf_file_names is done!')
    print(message_string)

    return normmalized_cubes_dict, netcdf_file_names


def normalize_mixed_cubes(NETCDF_HIGHRES_CUBES, normalization_dict):

    normmalized_cubes_dict = {}
    length = NETCDF_HIGHRES_CUBES[PREDICTOR_MATRIX_KEY].shape[0]

    for m in range(length):
        this_cube = NETCDF_HIGHRES_CUBES[PREDICTOR_MATRIX_KEY][m, :, :, :]
        normmalized_cubes_dict[m] = normalize_nam_maps(
        predictor_matrix = this_cube,
        predictor_names = NETCDF_HIGHRES_CUBES[PREDICTOR_NAMES_KEY],
        normalization_dict = normalization_dict)

    message_string = (
        'The normalization of ' + 'netcdf_file_names is done!')
    print(message_string)

    return normmalized_cubes_dict


def normalize_sst_cubes(NETCDF_HIGHRES_CUBES, normalization_dict):

    normmalized_cubes_dict = {}
    length = NETCDF_HIGHRES_CUBES[SST_MATRIX_KEY].shape[0]

    for m in range(length):
        this_cube = NETCDF_HIGHRES_CUBES[SST_MATRIX_KEY][m, :, :, :]
        normmalized_cubes_dict[m] = normalize_sst_map(
        predictor_matrix = this_cube,
        predictor_names = NETCDF_HIGHRES_CUBES[SST_NAME_KEY],
        normalization_dict = normalization_dict)

    message_string = (
        'The normalization of ' + 'netcdf_file_names is done!')
    print(message_string)

    return normmalized_cubes_dict


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
    fig, ax = pyplot.subplots(figsize = FIG_DEFULT_SIZE)
    ax = sns.barplot(x = "VIS-Cat", y="VIS_Count", data=df,
                     palette="Blues_d")
    xlocs, xlabs = pyplot.xticks()
    pyplot.xlabel('Visibility class')
    pyplot.ylabel('The number of cases')
    txt = ('The number of visibility cases for {0}').format(year)
    pyplot.title(txt)
    for i, v in enumerate(df["VIS_Count"]):
        pyplot.text(xlocs[i] , v + margin, str(v),
                    fontsize=12, color='red',
                    horizontalalignment='center', verticalalignment='center')

    pyplot.show()



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


#===============================================================================
#============================ Loading Cube Data ================================
#===============================================================================

def npz_cube_creator(year_information, target, DEFAULT_IMAGE_DIR_NAME):
    for key in year_information:

        year = key
        year_dir = DEFAULT_IMAGE_DIR_NAME + year + '/'
        first_date_string, last_date_string = year_information[key]

        nam_cubes_names, mur_cubes_names, targets = find_map_name_date(
            first_date_string= first_date_string, last_date_string=last_date_string, target = target, image_dir_name= year_dir)

        print(('The number of NAM cubes of {0}: ').format(key), len(nam_cubes_names))
        print(('The number of MUR cubes of {0}: ').format(key), len(mur_cubes_names))

        # save the test file into txt file:

        nam_map_names_file = 'NamFileNames' + key + '_24.txt'
        nam_map_names_path = os.path.join(DEFAULT_24HOURS_NAMES_DIR, nam_map_names_file)
        with open(nam_map_names_path, 'w') as filehandle:
            for listitem in nam_cubes_names:
                filehandle.write('%s\n' % listitem)

        # save the test file into txt file:

        mur_map_names_file = 'MurfileNames' + key + '_24.txt'
        mur_map_names_path = os.path.join(DEFAULT_24HOURS_NAMES_DIR, mur_map_names_file)
        with open(mur_map_names_path, 'w') as filehandle:
            for listitem in mur_cubes_names:
                filehandle.write('%s\n' % listitem)

        #save target file as .csv file
        target_file_name = 'target' + key + '_24.csv'
        target_file_path = os.path.join(DEFAULT_24HOURS_TARGET_DIR, target_file_name)
        targets.to_csv(target_file_path)


        print('The reading file of ' + key + ' is started!')
        # Select the cubes based on their hour predictions
        nam_concatcubes_names = find_nam_cubes_name_hourpredict(nam_cubes_names,
            hour_prediction_names = ['000', '006', '012','024'])
        print(('The number of cubes of {0}: ').format(key), len(nam_concatcubes_names))

        #reading MUR data
        Mur12cubes_names = []
        Mur12cubes_names = copy_mur_name_ntimes(mur_cubes_names, Mur12cubes_names, 16)
        print(('The number of MUR files per day (12 times) of {0}: ').format(key), len(Mur12cubes_names))


        NETCDF_MIXED_CUBES = mixed_cubes(nam_concatcubes_names, NETCDF_TMP_NAMES, Mur12cubes_names, NETCDF_MUR_NAMES)
        print(NETCDF_MIXED_CUBES['sst_matrix'].shape)

        print(NETCDF_MIXED_CUBES['predictor_matrix'].shape)
        print(NETCDF_MIXED_CUBES['predictor_names'])

        normalization_mixed_dict = get_mixed_normalization_params(NETCDF_MIXED_CUBES, NETCDF_GEN_NAMES)
        normalized_mixed_cubes = normalize_mixed_cubes(NETCDF_MIXED_CUBES, normalization_mixed_dict)
        #normalized_mixed_cubes = scaler_mixed_cubes(NETCDF_MIXED_CUBES, NETCDF_GEN_NAMES)
        cancat_mixed_cube = concate_mixed_cubes_files(normalized_mixed_cubes)
        print('The shape of Mix cube: ', cancat_mixed_cube.shape)
        mixed_cube_name = 'NETCDF_MIXED_CUBE_' + key + '_24.npz'
        mixed_cube_path = os.path.join(DEFAULT_24HOURS_CUBES_DIR, mixed_cube_name)
        savez_compressed(mixed_cube_path, cancat_mixed_cube)

        normalization_sst_dict = get_sst_normalization_params(NETCDF_MIXED_CUBES, NETCDF_MUR_NAMES)
        normalized_sst_cubes = normalize_sst_cubes(NETCDF_MIXED_CUBES, normalization_sst_dict)
        #normalized_sst_cubes = scaler_sst_cubes(NETCDF_MIXED_CUBES, NETCDF_MUR_NAMES)

        cancat_sst_cube = concate_sst_cubes_files(normalized_sst_cubes)
        print('The shape of SST cube: ', cancat_sst_cube.shape)

        sst_cube_name = 'NETCDF_SST_CUBE_' + key + '_24.npz'
        sst_cube_path = os.path.join(DEFAULT_24HOURS_CUBES_DIR, sst_cube_name)
        savez_compressed(sst_cube_path, cancat_sst_cube)

        # The NAM feature preparation:

        NAM_normalization_dict = get_nam_normalization_params(nam_concatcubes_names, NETCDF_PREDICTOR_NAMES['NewOrder'])
        normalized_nam_cubes = normalize_many_cubes(nam_concatcubes_names, NAM_normalization_dict, NETCDF_PREDICTOR_NAMES['NewOrder'])
        #normalized_nam_cubes = scaler_many_cubes(nam_concatcubes_names, NETCDF_PREDICTOR_NAMES['Physical_G4'])
        nam_concat_cubes = concate_nam_cubes_files(normalized_nam_cubes, NETCDF_PREDICTOR_NAMES['NewOrder'])

        nam_cube = nam_cubes_dict(nam_concat_cubes)
        print("The trainig cube size of whole cube: ", nam_cube['predictor_matrix'].shape)

        final_nam_cube = nam_cube['predictor_matrix']
        print("The trainig cube size of whole cube: ", final_nam_cube.shape)

        nam_cube_name = 'NETCDF_NAM_CUBE' + key + '_24.npz'
        nam_cube_path = os.path.join(DEFAULT_24HOURS_CUBES_DIR, nam_cube_name)
        savez_compressed(nam_cube_path, final_nam_cube)

        print('The iteration is done!')




def load_cube(name):
    load_yeras = load(name)
    load_yeras = load_yeras['arr_0']
    return load_yeras

def concat_npz_cube_files(nam_file_names, mixed_file_names, mur_file_names, root, years):

    NAM_CUBE   = None
    SST_CUBE   = None
    MIXED_CUBE = None


    for i in years:
        nam_name = root + nam_file_names[i]
        load_nam_cube = load_cube(nam_name)
        #print('The size of nam training input cube with float64 format: ', getsizeof(nam_cube))
        this_nam_cube = numpy.float32(load_nam_cube)
       #print('The size of nam training input cube with float32 format: ', getsizeof(nam_cube))
        #print()
        #print(nam_cube.shape)

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


        if NAM_CUBE is None:
            NAM_CUBE = this_nam_cube
        else:
            NAM_CUBE = numpy.concatenate((NAM_CUBE, this_nam_cube), axis = 0)

        #print('Training NAM input size: ', training_nam_cube.shape)
        del load_nam_cube, this_nam_cube
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



    NAM_CUBES = numpy.concatenate((MIXED_CUBE, NAM_CUBE), axis = -1)
    #print('Training final NAM input size: ', training_cube.shape)
    #print('Training final MUR input size: ', training_mur_cube.shape)
    #print()
    del MIXED_CUBE, NAM_CUBE


    return NAM_CUBES, SST_CUBE


def load_cube_data(nam_file_names, mixed_file_names, mur_file_names, file_names_root,
    training_years, validation_years, testing_years):
    print("======================= Training Input Size=====================")
    Xtrain_nam, Xtrain_mur = concat_npz_cube_files(
    nam_file_names, mixed_file_names, mur_file_names, file_names_root, training_years)
    Xtrain_nam = numpy.expand_dims(Xtrain_nam, axis = -1)
    Xtrain_mur = numpy.expand_dims(Xtrain_mur, axis = -1)
    print('Training final NAM input size: ', Xtrain_nam.shape)
    print('Training final MUR input size: ', Xtrain_mur.shape)


    print("====================== Validation Input Size====================")
    Xvalid_nam, Xvalid_mur = concat_npz_cube_files(
    nam_file_names, mixed_file_names, mur_file_names, file_names_root, validation_years)
    Xvalid_nam = numpy.expand_dims(Xvalid_nam, axis = -1)
    Xvalid_mur = numpy.expand_dims(Xvalid_mur, axis = -1)
    print('Validation final NAM input size: ', Xvalid_nam.shape)
    print('Validation final MUR input size: ', Xvalid_mur.shape)


    print("======================= testing Input Size======================")
    Xtest_nam, Xtest_mur = concat_npz_cube_files(
    nam_file_names, mixed_file_names, mur_file_names, file_names_root, testing_years)
    Xtest_nam = numpy.expand_dims(Xtest_nam, axis = -1)
    Xtest_mur = numpy.expand_dims(Xtest_mur, axis = -1)
    print('Training final NAM input size: ', Xtest_nam.shape)
    print('Training final MUR input size: ', Xtest_mur.shape)
    print("================================================================")



    return Xtrain_nam, Xtrain_mur, Xvalid_nam, Xvalid_mur, Xtest_nam, Xtest_mur


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







def load_Cat_cube_data_sh(shuffled_idx, nam_G1_names, nam_G2_names, nam_G3_names,nam_G4_names, mixed_file_names, mur_file_names, file_names_root, years):
    X_Nam_G1, X_Nam_G2, X_Nam_G3, X_Nam_G4, X_mixed, X_mur = concat_cat_npz_cube_files(
        nam_G1_names, nam_G2_names, nam_G3_names,nam_G4_names, mixed_file_names, mur_file_names, file_names_root, years)


    #print(shuffled_idx)
    X_Nam_G3[:] = X_Nam_G3[:,:,:, shuffled_idx]
    X_Nam_G3 = tensorflow.convert_to_tensor(X_Nam_G3)



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


def sample_data():
    #import datagenerator
    nam_file_names = ['NETCDF_NAM_CUBE_2020.npz']

    mixed_file_names = ['NETCDF_MIXED_CUBE_2020.npz']

    mur_file_names = ['NETCDF_SST_CUBE_2020.npz']

    file_names_root = './Dataset/INPUT/'

    nam_name =  file_names_root + nam_file_names[0]
    nam_cube = load_cube(nam_name)
    nam_cube = numpy.float32(nam_cube)

    mixed_name =  file_names_root + mixed_file_names[0]
    mixed_cube = load_cube(mixed_name)
    mixed_cube = numpy.float32(mixed_cube)

    mur_name =  file_names_root + mur_file_names[0]
    mur_cube = load_cube(mur_name)
    Xtrain_mur = numpy.float32(mur_cube)

    Xtrain_nam = numpy.concatenate((mixed_cube, nam_cube), axis = -1)




    print("======================= Data Input Size=====================")
    Xtrain_nam = numpy.expand_dims(Xtrain_nam, axis = -1)
    Xtrain_mur = numpy.expand_dims(Xtrain_mur, axis = -1)
    print('Training final NAM input size: ', Xtrain_nam.shape)
    print('Training final MUR input size: ', Xtrain_mur.shape)


    Training_targets_name = '{0:s}/target2020.csv'.format(DEFAULT_TARGET_DIR_NAME)
    Training_targets_file = pandas.read_csv(Training_targets_name, header=0, sep=',')
    Training_targets = Training_targets_file['VIS_Cat']
    ytrain = to_categorical(Training_targets)
    print("Size of training targets: ", ytrain.shape)

    return Xtrain_nam, Xtrain_mur, Training_targets, ytrain





