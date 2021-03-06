# Binarize fog predictions
# using a threshold
# The optimal threshold is calculated after training

import numpy as np
import pandas as pd
import os
from optparse import OptionParser

def binarize(probs, threshold):
    probs = probs[:,1]
    classes = [0]*len(probs)
    yes = probs > threshold
    classes = classes + yes
    classes = classes.astype("int")
    return classes


def main():
    parser = OptionParser()
    parser.add_option("-p", "--predictions",
                      help="Path to probabilistic predictions csv with colums 'pred_fog', 'pred_non'.")
    parser.add_option("-t", "--threshold",
                      help="Threshold for converting fog probability to fog classification  [default = %default].",
                      default=0.129, type="float")
    parser.add_option("-o", "--output",
                      help="Path to save the output binary predictions csv.")
    (options, args) = parser.parse_args()

    predFile = options.predictions
    outFile = options.output
    threshold = options.threshold

    if predFile is None:
        print("Must specify an input predictions file (-p).\nExiting...")
        exit(1)
    if not os.path.exists(predFile):
        print("Input predictions file {} not found.\nExiting...".format(predFile))
        exit(1)

    if outFile is None:
        print("Must specify output file to save binary predictions (-o).\nExiting...")
        exit(1)

    # Load predicted probabilities
    dfPreds = pd.read_csv(predFile)
    preds = np.vstack(dfPreds.to_numpy())
    # Binarize with threshold
    classes = binarize(preds, threshold)
    # Count
    numNonFog = np.count_nonzero(classes)
    # Save (with added binary column)
    classes = np.reshape(classes, (len(classes), 1))
    preds = np.concatenate((preds, classes), axis=1)

    dfPreds["pred_class"] = classes
    dfPreds["pred_className"] = ["fog" if c == 0 else "non-fog" for c in classes]
    dfPreds.to_csv(outFile, index=False)

    print("")
    print("Predictions file: {}".format(predFile))
    print("Threshold = {}".format(threshold))
    print("")
    print("Total cases: {}".format(len(preds)))
    print("Fog cases: {}".format(len(preds) - numNonFog))
    print("Non-fog cases: {}".format(numNonFog))
    print("")
    print("Output file: {}".format(outFile))
    print("")


if __name__ == "__main__":
    main()
