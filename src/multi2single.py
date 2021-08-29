###############################################
#            No longer needed!                #
# Since correctly saving model after training #
###############################################

# Convert a multi-GPU model to single-GPU model
# To clarify, the FogNet training script outputs model weights that only work with a multi-GPU model
# This is very inconvenient for prediction. For example, the batch size must be divisible by the number of GPUs.
import tensorflow
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
from optparse import OptionParser

# Fognet modules
import FogNet
import utils

parser = OptionParser()
parser.add_option("-w", "--weights",
                  help="Path to trained model weights")
parser.add_option("-d", "--directory",
                  help="Path to directory with fog data cubes")
parser.add_option("-l", "--label",
                  help="Unique label to identify data cubes in data directory")
parser.add_option("-t", "--time_horizon",
                  help="Prediction time horizon")
parser.add_option(      "--filters",
                  help="Number of filters [default = %default].",
                  default=24, type="int")
parser.add_option(      "--dropout",
                  help="Droput rate [default = %default].",
                  default=0.3, type="float")
parser.add_option("-o", "--output_weights",
                  help="Path to file to save single-GPU weights")
(options, args) = parser.parse_args()

inputWeightsFile = options.weights
outputWeightsFile = options.output_weights
dataDir = options.directory
dataLabel = options.label
dataTime = options.time_horizon
filters = options.filters
dropout = options.dropout

nGPU = 4

# Global file path templates
nam_G1_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG1_{horizon}.npz"
nam_G2_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG2_{horizon}.npz"
nam_G3_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG3_{horizon}.npz"
nam_G4_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG4_{horizon}.npz"
mixed_file_template = "{dir}/NETCDF_MIXED_CUBE_{label}_{horizon}.npz"
mur_file_template = "{dir}/NETCDF_SST_CUBE_{label}.npz"

def loadCubes(dataDir, dataLabel, dataTime):
# Load data cubes
    dir="/"
    nam_G1_file = [nam_G1_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G2_file = [nam_G2_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G3_file = [nam_G3_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G4_file = [nam_G4_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    mixed_file  = [mixed_file_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    mur_file    = [mur_file_template.format(dir=dir, label=dataLabel)]
    cubes = utils.load_Cat_cube_data(
        nam_G1_file, nam_G2_file, nam_G3_file, nam_G4_file, mixed_file, mur_file,
        dataDir, [0])

    cubeShapes = [cubes[0].shape[1:],
                  cubes[1].shape[1:],
                  cubes[2].shape[1:],
                  cubes[3].shape[1:],
                  cubes[4].shape[1:],
                  cubes[5].shape[1:]]
    return cubes, cubeShapes


# Determine model shapes from data shapes
cubes, cubeShapes = loadCubes(dataDir, dataLabel, dataTime)

# Init model
C = FogNet.FogNet(
    Input(cubeShapes[0]),
    Input(cubeShapes[1]),
    Input(cubeShapes[2]),
    Input(cubeShapes[3]),
    Input(cubeShapes[4]),
    Input(cubeShapes[5]),
    filters,
    dropout,
    2
)

model = C.BuildModel()
#model = multi_gpu_model(model, gpus=nGPU)
load_status = model.load_weights(inputWeightsFile)
#old_model = model.layers[-2]
#old_model.save(outputWeightsFile)
