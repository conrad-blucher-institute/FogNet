# Given predictions and targets,
# Calculate model skill
# and categorize each instance

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-p", "--predictions",
                      help="Path to binarized predictions. CSV with columns 'pred_fog', 'pred_non', 'pred_class', 'pred_className'")
    parser.add_option("-t", "--targets",
                      help="Path to targets. Each line is either 0 (fog) or 1 (non-fog).")
    parser.add_option("-o", "--output",
                      help="Path to save the output binary predictions")
    (options, args) = parser.parse_args()

    predFile = options.predictions
    targetFile = options.targets
    outFile = options.output

    if predFile is None:
        print("Must specify an input predictions file (-p).\nExiting...")
        exit(1)
    if not os.path.exists(predFile):
        print("Input predictions file {} not found.\nExiting...".format(predFile))
        exit(1)

    if targetFile is None:
        print("Must specify an input targets file (-t).\nExiting...")
        exit(1)
    if not os.path.exists(targetFile):
        print("Input targets file {} not found.\nExiting...".format(targetFile))
        exit(1)

    if outFile is None:
        print("Must specify output file to save binary predictions (-o).\nExiting...")
        exit(1)


    # Load data
    dfPreds = pd.read_csv(predFile)
    dfTargets = pd.read_csv(targetFile, header=None)
    targets = dfTargets.iloc[:, 0].values
    dfPreds["target_class"] = targets
    dfPreds["target_className"] = ["fog" if c == 0 else "non-fog" for c in targets]

    # Calc metrics
    y = dfPreds["target_class"]
    y_pred = dfPreds["pred_class"]

    error     = ["" for i in range(len(y))]
    errorType = ["" for i in range(len(y))]

    for i in range(len(y)):
        p = y[i]       # pred
        t = y_pred[i]  # target

        # Hit
        if p == 0 and t == 0:
            error[i] = "a"
            errorType[i] = "hit"
        # False alarm
        elif p == 0 and t == 1:
            error[i] = "b"
            errorType[i] = "false-alarm"
        # Miss
        elif p == 1 and t == 0:
            error[i] = "c"
            errorType[i] = "miss"
        # Correct rejection
        elif p == 1 and t == 1:
            error[i] = "d"
            errorType[i] = "correct-reject"

    dfPreds["outcome"] = error
    dfPreds["outcome_name"] = errorType

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    a = tn    # Hit
    b = fn    # false alarm
    c = fp    # miss
    d = tp    # correct rejection

    POD  = a/(a+c)
    F    = b/(b+d)
    FAR  = b/(a+b)
    CSI  = a/(a+b+c)
    PSS  = ((a*d)-(b*c))/((b+d)*(a+c))
    HSS  = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
    ORSS = ((a*d)-(b*c))/((a*d)+(b*c))
    CSS  = ((a*d)-(b*c))/((a+b)*(c+d))

    print("")
    print("Hits:               {}".format(a))
    print("False alarms:       {}".format(b))
    print("Misses:             {}".format(c))
    print("Correct rejections: {}".format(d))
    print("------")
    print("POD:  {}".format(POD))
    print("F:    {}".format(F))
    print("FAR:  {}".format(FAR))
    print("CSI:  {}".format(CSI))
    print("PSS:  {}".format(PSS))
    print("HSS:  {}".format(HSS))
    print("ORSS: {}".format(ORSS))
    print("CSS:  {}".format(CSS))
    print("")

    print("Added outcome to each CSV row: {}".format(outFile))
    print("")

    dfPreds.to_csv(outFile, index=False)

if __name__ == "__main__":
    main()
