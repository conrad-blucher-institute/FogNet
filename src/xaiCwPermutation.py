import numpy as np
import pandas as pd
import pickle
from optparse import OptionParser
import time
import copy
import gc
import os
from sklearn.metrics import confusion_matrix

# FogNet imports
import eval

def evalPFI(cubesArray, cubes, model, sst_cube, storage):
    pS = 0
    pN = cubes[0].shape[3]
    storage[0][...] = np.reshape(cubesArray[:,:,:,pS:pN],
        (cubesArray.shape[0], cubes[0].shape[1], cubes[0].shape[2], cubes[0].shape[3], cubes[0].shape[4]))
    pS = pN
    pN = pN + cubes[1].shape[3]
    storage[1][...] = np.reshape(cubesArray[:,:,:,pS:pN],
        (cubesArray.shape[0], cubes[1].shape[1], cubes[1].shape[2], cubes[1].shape[3], cubes[1].shape[4]))
    pS = pN
    pN = pN + cubes[2].shape[3]
    storage[2][...] = np.reshape(cubesArray[:,:,:,pS:pN],
        (cubesArray.shape[0], cubes[2].shape[1], cubes[2].shape[2], cubes[2].shape[3], cubes[2].shape[4]))
    pS = pN
    pN = pN + cubes[3].shape[3]
    storage[3][...] = np.reshape(cubesArray[:,:,:,pS:pN],
        (cubesArray.shape[0], cubes[3].shape[1], cubes[3].shape[2], cubes[3].shape[3], cubes[3].shape[4]))
    pS = pN
    pN = pN + cubes[4].shape[3]
    storage[4][...] = np.reshape(cubesArray[:,:,:,pS:pN],
        (cubesArray.shape[0], cubes[4].shape[1], cubes[4].shape[2], cubes[4].shape[3], cubes[4].shape[4]))
    pS = pN
    pN = pN + cubes[5].shape[3]

    #storage[5][...] = np.reshape(cubesArray[:,:,:,pS:pN],
    #    (cubesArray.shape[0], cubes[5].shape[1], cubes[5].shape[2], cubes[5].shape[3], cubes[5].shape[4]))

    # Temp hack for SST
    storage[5][...] = sst_cube

    # Return prediction
    return eval.evalModel(model, storage)


def calcPerformance(y, ypred, threshold = None):

    # Apply threshold -> binarize predictions
    ypred_ = ypred[:, 1] > threshold
    ypred_ = ypred_.astype(int)

    #  Calculate confusion matrix to calculate performance metrics
    tn, fp, fn, tp = confusion_matrix(y, ypred_, labels=[0, 1]).ravel()
    a = tn # Hit
    b = fn # False alarm
    c = fp # Miss
    d = tp # Correct rejection

    # Calculate performance metrics
    HSS = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
    return HSS

def main():

    parser = OptionParser()
    # Save results
    parser.add_option("-o", "--output_pfi_values",
                      help="Path to save output PFI values (.npz).")
    # Load model & data
    parser.add_option("-k", "--targets",
                      help="Path to targets. Each line is either 0 (fog) or 1 (non-fog).",
                      default="trained_model/test_targets.txt")
    parser.add_option("-w", "--weights",
                      help="Path to trained model weights [default = %default].",
                      default="trained_weights.h5")
    parser.add_option("-d", "--directory",
                      help="Path to directory with fog data cubes [default = %default].",
                      default="/data1/fog/fognn/Dataset/24HOURS/INPUT/")
    parser.add_option("-l", "--labels",
                      help="Unique (comma-delimited) label(s) to identify data cubes in data directory [default = %default].",
                      default="2018,2019,2020")
    parser.add_option("-t", "--time_horizon",
                      help="Prediction time horizon [default = %default].",
                      default="24")
    parser.add_option("-i", "--filters",
                      help="Number of filters [default = %default].",
                      default=24, type="int")
    parser.add_option("-j", "--dropout",
                      help="Droput rate [default = %default].",
                      default=0.3, type="float")
    parser.add_option("-c", "--cases",
                      help="Path to list of indices in data cube to evaluate. If none, use all.")
    parser.add_option(      "--threshold",
                      help="Threshold for converting fog probability to fog classification  [default = %default].",
                      default=0.129, type="float")
    # PermutationFeatureImportance parameters
    parser.add_option("-r", "--repeats",
                      help="Number of PFI repetitions [default = %default].",
                      default=2, type="int")
    (options, args) = parser.parse_args()

    # Data options
    targetFile  = options.targets
    weightsFile = options.weights
    dataDir     = options.directory
    dataLabels  = options.labels.split(",")
    dataTime    = options.time_horizon
    casesFile   = options.cases
    # Architecture params (must match the weights!)
    filters = options.filters
    dropout = options.dropout
    threshold = options.threshold
    # Output
    outFile = options.output_pfi_values
    # PermutationFeatureImportance options
    repeats = options.repeats

    if outFile is None:
        print("Expected output file (-o).\nExiting...")
        exit(1)

    if targetFile is None:
        print("Expected target file (-k), since PFI is based on change in performance.\nExiting...")
        exit(1)
    if not os.path.exists(targetFile):
        print("Input targets file {} not found.\nExiting...".format(targetFile))
        exit(1)

    dfTargets = pd.read_csv(targetFile, header=None)
    targets = np.array(dfTargets[0])

    cases = None
    if casesFile is not None:
        cases = np.loadtxt(casesFile, comments=['#', '$', '@']).astype("int")

    # Load data
    cubesList = [None for dL in dataLabels]
    for i, dL in enumerate(dataLabels):
        cubesList[i], cubeShapes = eval.loadCubes(dataDir, dL, dataTime)

    cubes = [
        np.vstack([cubesList[i][0] for i in range(len(cubesList))]),
        np.vstack([cubesList[i][1] for i in range(len(cubesList))]),
        np.vstack([cubesList[i][2] for i in range(len(cubesList))]),
        np.vstack([cubesList[i][3] for i in range(len(cubesList))]),
        np.vstack([cubesList[i][4] for i in range(len(cubesList))]),
        np.vstack([cubesList[i][5] for i in range(len(cubesList))]),
    ]

    # Subset data to selected cases
    if cases is not None:
        cubes_ = copy.deepcopy(cubes)
        cubes_[0] = cubes[0][cases]
        cubes_[1] = cubes[1][cases]
        cubes_[2] = cubes[2][cases]
        cubes_[3] = cubes[3][cases]
        cubes_[4] = cubes[4][cases]
        cubes_[5] = cubes[5][cases]
        cubes = cubes_

        targets = targets[cases]

    # Initialize model
    model = eval.initModel(cubeShapes, weightsFile, filters, dropout)

    # Calculate base model performance
    preds = eval.evalModel(model, cubes)
    base_HSS = calcPerformance(targets, preds, threshold = threshold)
    print("Base model HSS = {}".format(base_HSS))

    # Convert from list of groups to single numpy array
    # The last channel has a different size. The following is a hack to have a single rows x cols x bands array
    # Replace it with a 32 x 32 band of (for now) arbitrary values
    sst_cube = cubes[-1]
    cubes[-1] = np.ones((cubes[0].shape[0], 32, 32, 1, 1))
    # Concatenate list of groups into single rows x cols x bands array
    cubesArray = np.squeeze(np.concatenate(cubes, axis = 3))

    # Get data shape
    samples, rows, cols, bands = cubesArray.shape

    # Define superpixel mask params
    mask_shape = (8, 8)
    nMasks = int(((rows / mask_shape[0]) * (cols / mask_shape[1])) * bands)

    # Generate a set of superpixel masks that are applied to each channel
    x, y = rows // mask_shape[0], cols // mask_shape[1]
    mask_set = np.zeros((x * y, rows, cols), dtype=bool)
    count = 0
    for i in range(x):
        for j in range(y):
            mask_set[count][i*mask_shape[0]:i*mask_shape[0]+mask_shape[0], j*mask_shape[1]:j*mask_shape[1]+mask_shape[1]] = 1
            count += 1

    # Init storage of PFI results, each repeat separately
    pfi_percents_reps = np.zeros((repeats, rows, cols, bands))

    # Initialize temp data storage
    rands = np.zeros((samples, rows,  cols))
    band_backup = np.zeros(cubesArray[:, :, :, 0].shape)
    preds = np.zeros((samples, 2))
    storage = [
        np.zeros(cubes[0].shape),
        np.zeros(cubes[1].shape),
        np.zeros(cubes[2].shape),
        np.zeros(cubes[3].shape),
        np.zeros(cubes[4].shape),
        np.zeros(sst_cube.shape),
    ]

    # Perform PFI
    start_time = time.time()
    for r in range(repeats):
        for b in range(bands):
            # Backup original bands
            band_backup[...] = cubesArray[:, :, :, b]

            for m, mask in enumerate(mask_set):
                # Replace superpixel with random values
                # TODO: permute existing instead of random to match Hamid's
                #rands[...] = np.random.random(size=(samples, rows, cols))
                rands[...] = np.random.permutation(band_backup)
                cubesArray[:, :, :, b] = rands * mask + cubesArray[:, :, :, b] * np.invert(mask)

                # Evaluate model with permuted superpixel
                preds[...] = evalPFI(cubesArray, cubes, model, sst_cube, storage)

                # Revert band to original data
                cubesArray[:, :, :, b] = band_backup

                # Calculate loss (HSS)
                HSS = calcPerformance(targets, preds, threshold = threshold)

                # Store loss
                pfi_percents_reps[r, :, :, b] += HSS * mask

                # Clean up memory to get ready for next loop
                gc.collect()

                print("Mask {} / {}".format(m, mask_set.shape[0]))
            print("Repeat {}, channel {}, elapsed time: {} seconds.".format(r + 1, b, time.time() - start_time))
    end_time = time.time()

    # Average losses from the repetitions
    pfi_percents = np.mean(pfi_percents_reps, axis=0)

    # Calculate PFI: average permuted feature loss - original model loss
    pfi_values = pfi_percents - base_HSS

    # Save PFI values
    np.savez_compressed(outFile, pfi_values,)


if __name__ == '__main__':
    main()
