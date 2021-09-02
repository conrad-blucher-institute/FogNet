# This script runs a trained Fognet model
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import os
import numpy as np
import pandas as pd
from optparse import OptionParser
from tensorflow.keras.utils import multi_gpu_model

# Fognet modules
import FogNet
import utils

# Global file path templates
nam_G1_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG1_{horizon}.npz"
nam_G2_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG2_{horizon}.npz"
nam_G3_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG3_{horizon}.npz"
nam_G4_template = "{dir}/NETCDF_NAM_CUBE_{label}_PhG4_{horizon}.npz"
mixed_file_template = "{dir}/NETCDF_MIXED_CUBE_{label}_{horizon}.npz"
mur_file_template = "{dir}/NETCDF_SST_CUBE_{label}.npz"

def printTemplate(dir="<DIR>", label="<LABEL>", time="<TIME>"):
    print("  NAM G1: " + nam_G1_template.format(dir=dir, label=label, horizon=time))
    print("  NAM G2: " + nam_G2_template.format(dir=dir, label=label, horizon=time))
    print("  NAM G3: " + nam_G3_template.format(dir=dir, label=label, horizon=time))
    print("  NAM G4: " + nam_G4_template.format(dir=dir, label=label, horizon=time))
    print("  Mixed:  " + mixed_file_template.format(dir=dir, label=label, horizon=time))
    print("  SST:    " + mur_file_template.format(dir=dir, label=label))

def validateOptions(weightsFile, dataLabel, dataTime, dataDir):

    # Weights
    if weightsFile is None:
        print("Must specify a trained FogNet weights file (-w)")
        print("Exiting...")
        exit(1)

    hasDataFiles = True

    # Data label
    if dataLabel is None:
        print("Must specify a label (-l) to uniquely access FogNet data cubes in data directory")
        hasDataFiles = False

    # Data time horizon
    if dataTime is None:
        print("Must specify a time horizon (-t) to access FogNet data cubes in data director")
        hasDataFiles = False

    # Data directory
    if dataDir is None:
        print("Must specify a directory (-d) with FogNet data cube files")
        hasDataFiles = False
    if not os.path.isdir(dataDir):
        print("Cannot find FogNet data directory: {}".format(dataDir))
        print("Expecting it to contain:")
        hasDataFiles = False

    # Print example data cubes
    if hasDataFiles == False:
        print("Required to have data in format:")
        printTemplate()
        print("Exiting...")
        exit(0)

def loadCubes(dataDir, dataLabel, dataTime):
# Load data cubes
    dir="/"
    nam_G1_file = [nam_G1_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G2_file = [nam_G2_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G3_file = [nam_G3_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    nam_G4_file = [nam_G4_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    mixed_file  = [mixed_file_template.format(dir=dir, label=dataLabel, horizon=dataTime)]
    mur_file    = [mur_file_template.format(dir=dir, label=dataLabel)]

    cubes = utils.load_Cat_cube_data(nam_G1_file,
        nam_G2_file, nam_G3_file, nam_G4_file, mixed_file, mur_file,
        dataDir, [0])

    nam_G1_shape = cubes[0].shape[1:]
    nam_G2_shape = cubes[1].shape[1:]
    nam_G3_shape = cubes[2].shape[1:]
    nam_G4_shape = cubes[3].shape[1:]
    mixed_shape  = cubes[4].shape[1:]
    mur_shape    = cubes[5].shape[1:]

    cubeShapes = [nam_G1_shape,
                  nam_G2_shape,
                  nam_G3_shape,
                  nam_G4_shape,
                  mixed_shape,
                  mur_shape]

    return cubes, cubeShapes


def initModel(cubeShapes, weightsFile, filters=24, dropout=0.3):
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
        2)
    model = C.BuildModel()
    #model = multi_gpu_model(model, gpus=4)
    # Loads weights
    load_status = model.load_weights(weightsFile)

    return model


def evalModel(model, cubes):
    # Evaluate model
    return  model.predict(cubes)


def writePreds(preds, outFile=None):
    if outFile is None:
        # Print output to screen
        for i, p in enumerate(preds):
            print("{}:  {}".format(i, p))
    else:
        # Save output to csv
        dfPreds = pd.DataFrame(preds, columns=["pred_fog", "pred_non"])
        dfPreds.to_csv(outFile, index=False)


def main():
    parser = OptionParser()
    parser.add_option("-w", "--weights",
                      help="Path to trained model weights.")
    parser.add_option("-d", "--directory",
                      help="Path to directory with fog data cubes.")
    parser.add_option("-l", "--label",
                      help="Comma-delimited list of unique labels to identify data cubes in data directory.")
    parser.add_option("-t", "--time_horizon",
                      help="Prediction time horizon.")
    parser.add_option("-o", "--output_predictions",
                      help="Path to file to save predictions csv.")
    parser.add_option(      "--filters",
                      help="Number of filters [default = %default].",
                      default=24, type="int")
    parser.add_option(      "--dropout",
                      help="Droput rate [default = %default].",
                      default=0.3, type="float")
    parser.add_option("-v", "--verbose",
                      help="Verbose output.",
                      default=False, action="store_true")
    (options, args) = parser.parse_args()

    # Data options
    weightsFile = options.weights
    dataDir = options.directory
    dataLabels = options.label.split(",")
    dataTime = options.time_horizon

    # Architecture params
    filters = options.filters
    dropout = options.dropout

    # Output options
    outFile = options.output_predictions

    # Print options
    verbose = options.verbose

    # Print information
    if verbose:
        print("Running FogNet")
        print("--------------")
        print("Model: {}".format(weightsFile))
        print("Params: ")
        print("  Num. filters: {}".format(filters))
        print("  Dropout: {}".format(dropout))
        print("Data cubes:")
        printTemplate(dir=dataDir, label=dataLabel, time=dataTime)

    preds_ = []

    for dataLabel in dataLabels:
        # Load data cubes from files
        cubes, cubeShapes = loadCubes(dataDir, dataLabel, dataTime)

        # Initialize model
        model = initModel(cubeShapes, weightsFile, filters, dropout)

        # Get predictions
        preds = evalModel(model, cubes)

        # Concat
        preds_.append(preds)

    preds = np.vstack(preds_)

    # Output predictions
    writePreds(preds, outFile)

if __name__ == "__main__":
    main()

