import numpy as np
import pickle
from optparse import OptionParser
import shap

# FogNet imports
import eval

# Globals
replacementValue = 0

def main():

    parser = OptionParser()
    # Save results
    parser.add_option("-o", "--output_shap_values",
                      help="Path to save pickled SHAP values.")
    # Load model & data
    parser.add_option("-w", "--weights",
                      help="Path to trained model weights [default = %default].",
                      default="trained_weights.h5")
    parser.add_option("-d", "--directory",
                      help="Path to directory with fog data cubes [default = %default].",
                      default="/data1/fog/fognn/Dataset/24HOURS/INPUT/")
    parser.add_option("-l", "--label",
                      help="Unique label to identify data cubes in data directory [default = %default].",
                      default="2019")
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
    # PartitionShap parameters
    parser.add_option("-e", "--max_evaluations",
                      help="Maximum number of SHAP evaluations [default = %default].",
                      default=50000, type="int")
    parser.add_option("-m", "--masker",
                      help="Feature removal masking method. Examples are `blur=10,10` and `color=128`. The blur numbers must be positive. [default = %default].",
                      default="color=0.5")
    (options, args) = parser.parse_args()

    # Data options
    weightsFile = options.weights
    dataDir     = options.directory
    dataLabel   = options.label
    dataTime    = options.time_horizon
    casesFile   = options.cases
    # Architecture params (must match the weights!)
    filters = options.filters
    dropout = options.dropout
    # Output
    outFile = options.output_shap_values
    # PartitionShap options
    maxEvals = options.max_evaluations
    maskerOption = options.masker

    if outFile is None:
        print("Expected output file (-o).\nExiting...")
        exit(1)

    # Convert masker
    validMasker = True
    maskerDecompose = maskerOption.split("=")
    if maskerDecompose[0] == "blur":
        try:
            maskerDecompose = maskerDecompose[1].split(")")[0].split(",")
            num1, num2 = int(maskerDecompose[0]), int(maskerDecompose[1])
        except:
            validMasker = False
        maskerChoice = "blur({},{})".format(num1, num2)
    elif maskerDecompose[0] == "color":
        try:
            color = float(maskerDecompose[1])
            maskerChoice = "{}".format(color)
        except:
            validMasker = False
    else:
        validMasker = False

    if validMasker == False:
        print("Invalid masker `{}`\nExiting...".format(maskerOption))
        exit(1)

    cases = None
    if casesFile is not None:
        cases = np.loadtxt(casesFile).astype("int")

    # Load data
    cubes, cubeShapes = eval.loadCubes(dataDir, dataLabel, dataTime)

    # Format data for SHAP
    # The last channel has a different size. The following is a hack to have a single rows x cols x bands array
    # Save the last channel (SST map)
    cubesSST = cubes[-1]
    # Replace it with a 32 x 32 band of (for now) arbitrary values
    cubes[-1] = np.ones((cubes[0].shape[0], 32, 32, 1, 1))
    # Concatenate list of groups into single rows x cols x bands array
    cubesShap = np.squeeze(np.concatenate(cubes, axis = 3))

    # Subset cases
    if cases is not None:
        cubesShap = cubesShap[cases]
        cubesSST = cubesSST[cases]

    # If only one case, enforce a shape of (1, rows, cols, channels)
    # instead of the automatic change to (rows, cols, channels)
    if len(cubesShap.shape) == 3:
        cubesShap = np.expand_dims(cubesShap, axis=0)
        cubesSST = np.expand_dims(cubesSST, axis=0)

    # Make each cubeShap flag the index + 10
    # Later, will subtract the 10 to detect which instance is being evaluated
    for i in range(cubesShap.shape[0]):
        cubesShap[i,:,:,-1] = np.ones((32, 32)) * 10 + i

    # Initialize model
    model = eval.initModel(cubeShapes, weightsFile, filters, dropout)

    def evalShap(cubesShap):

        # Extract individual cubes from concat array
        pS = 0
        pN = cubes[0].shape[3]
        cube1 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[0].shape[1], cubes[0].shape[2], cubes[0].shape[3], cubes[0].shape[4]))
        pS = pN
        pN = pN + cubes[1].shape[3]
        cube2 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[1].shape[1], cubes[1].shape[2], cubes[1].shape[3], cubes[1].shape[4]))
        pS = pN
        pN = pN + cubes[2].shape[3]
        cube3 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[2].shape[1], cubes[2].shape[2], cubes[2].shape[3], cubes[2].shape[4]))
        pS = pN
        pN = pN + cubes[3].shape[3]
        cube4 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[3].shape[1], cubes[3].shape[2], cubes[3].shape[3], cubes[3].shape[4]))
        pS = pN
        pN = pN + cubes[4].shape[3]
        cube5 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[4].shape[1], cubes[4].shape[2], cubes[4].shape[3], cubes[4].shape[4]))

        pS = pN
        pN = pN + cubes[5].shape[3]
        cube6 = np.reshape(cubesShap[:,:,:,pS:pN], (cubesShap.shape[0], cubes[5].shape[1], cubes[5].shape[2], cubes[5].shape[3], cubes[5].shape[4]))

        # The following is a hack to deal with the problem
        # of the final channel having a different (row, col) shape
        # But SHAP needs a single (row, col, channels) array
        # Hack:
        #       1. Place a 32 x 32 dummy channel (cube6)
        #       2. Detect which pixels were replaced with the value V
        #       3. Access the actual 384 x 384 SST map
        #       4. Replace the superpixels that match up with the dummy replaced pixels
        #       5. But which is the right SST map?
        #           A. SHAP does not pass the index of which instance is being evaluated
        #           B. So, we add the instance ID to all cells of the dummy channel
        #           C. Get the ith SST where i is the ID stored in the dummy channel
        #           D. What if entire dummy was replaced?
        #                  Doesn't matter what the original SST was; will replace fully

        # Batch size
        N = cube1.shape[0]

        # Replacement value
        V = replacementValue

        # Scale from small to large SST
        scale = 12   # SST is (384 x 384), dummy is (32, 32) -->  384 / 32 = 12

        # Find which instance
        idx = np.max(cube6)
        found = False
        if idx >= 10:
            found = True
            idx = idx - 10

        # Get SST for this instance
        if found:
            bandSST = cubesSST[int(idx)]
        else:
            # Don't need the SST.. just replace all anyway
            bandSST = np.ones((384, 384, 1, 1)) * V
        # Replicate for each in batch
        cube6_ = np.array([bandSST for b in range(N)])

        # For each instance, detect and apply value replacement masker
        for i in range(N):
            smallData = cube6[i,:,:,0,0]
            largeData = cube6_[i,:,:,0,0]

            # Detect all (row, col) where the value was replaces
            rows, cols = np.where(smallData == V)
            smallRep = np.vstack((rows, cols)).T

            # Convert to large superpixel
            largeRep = smallRep * scale
            for p in largeRep:
                largeData[p[0]:p[0]+scale, p[1]:p[1]+scale] = V

            # Copy over to cube
            cube6_[i,:,:,0,0] = largeData

        # Combine all into FogNet input list of rasters
        cubes_ = [cube1, cube2, cube3, cube4, cube5, cube6_]

        # Return prediction
        return eval.evalModel(model, cubes_)

    # Evaluate SHAP
    classes = ["0",  # Fog
               "1"   # Not fog
               ]
    try:
        # Masker is of form 'color=NUMBER'
        replacementValue = float(maskerChoice)
        maskerChoice = np.zeros_like(cubesShap[0]) + replacementValue
    except:
        print("Currently only supporing value maskers of form 'color=value'")
        exit(0)

    masker = shap.maskers.Image(maskerChoice, cubesShap[0].shape, partition_scheme=1)
    explainer = shap.Explainer(evalShap, masker, output_names=classes)
    shap_values = explainer(cubesShap, max_evals=maxEvals, batch_size=64)

    # Save shap values
    with open(outFile, 'wb') as f:
        pickle.dump(shap_values, f)


if __name__ == '__main__':
    main()
