import numpy as np
import pickle
from optparse import OptionParser
import shap

# FogNet imports
import eval

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
    # ISSUE: Cannot include group 5 since we concat all cubes and its shape is different
    cubesShap = np.squeeze(np.concatenate(cubes[:-1], axis = 3))

    # Subset cases
    if cases is not None:
        cubesShap = cubesShap[cases]

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

        cubes_ = [cube1, cube2, cube3, cube4, cube5,
                  np.zeros((cubesShap.shape[0], 384, 384, 1, 1))      # Hack: cube 6 is a blank channel
                  ]

        return eval.evalModel(model, cubes_)

    # Evaluate SHAP
    classes = ["0",  # Fog
               "1"   # Not fog
               ]
    try:
        # Masker is of form 'color=NUMBER'
        maskerChoice = np.zeros_like(cubesShap[0]) + float(maskerChoice)
    except:
        # Masker is of form 'blur=NUMBER,NUMBER'
        pass
    masker = shap.maskers.Image(maskerChoice, cubesShap[0].shape, partition_scheme=1)
    explainer = shap.Explainer(evalShap, masker, output_names=classes)
    shap_values = explainer(cubesShap, max_evals=maxEvals, batch_size=64)

    # Save shap values
    with open(outFile, 'wb') as f:
        pickle.dump(shap_values, f)


if __name__ == '__main__':
    main()
