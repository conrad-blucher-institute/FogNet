import pandas as pd
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import errno
import glob
import time
import calendar
import json
import pickle
import random
import netCDF4
from numpy import savez_compressed
from optparse import OptionParser
from scipy.interpolate import (UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
from sklearn.metrics import confusion_matrix

# This script calculates group-wise SHAP values
# Given a set of pre-computed model outputs
# By separating the model outputs out,
# it is possible to modify the SHAP code and regenerate
# to customize how SHAP is calculated

def main():

    ###########
    # Options #
    ###########
    parser = OptionParser()

    # Output file
    # -----------
    # SHAP values for each group
    parser.add_option("-o", "--output",
        default="groupwise_shap_results.csv",
        help="Path to save SHAP values")

    # Outcomes file
    # -------------
    # Used to separate the SHAP values by outcome
    # Must have same number of lines as number of
    # FogNet cases used to generate SHAP results
    #
    # Example:
    #   `$ head -n 5 outcomes_2018-2019-2020.csv`
    #   outcome
    #   correct-reject
    #   correct-reject
    #   correct-reject
    #   correct-reject
    parser.add_option("-f", "--outcomes",
        default="outcomes_2018-2019-2020.csv",
        help="Path to FogNet outcome classes")

    # Precomputed values directory
    # ----------------------------
    # SHAP is calculated based on model outputs
    # These model outputs have been previously
    # generated and stored
    #
    # Example:
    #   `tree SHAP | head -n 15`
    #   SHAP
    #   ├── G1
    #   │   ├── history.h5
    #   │   ├── loss.png
    #   │   ├── run_testing_0_accuracy_report.txt
    #   │   ├── run_testing_0_report.txt
    #   │   ├── run_training_0_report.txt
    #   │   ├── run_validation_0_report.txt
    #   │   ├── VIS_Prob_TEST.csv
    #   │   ├── VIS_Prob_TRAIN.csv
    #   │   ├── VIS_Prob_VALID.csv
    #   │   └── weights.h5
    #   ├── G1G2
    #   │   ├── history.h5
    #   │   ├── loss.png
    parser.add_option("-s", "--shap_dir",
        default="/data1/fog/Hamid/old-codes/Hamid/FogNetImport/SHAP/",
        help="Path to directory with SHAP-based model outputs")

    (options, args) = parser.parse_args()

    output_file = options.output
    outcomes_file = options.outcomes
    csv_files_dir = options.shap_dir

    def calc_mean_prob_df(id1, id2, idxs=None):
        folders = sorted(os.listdir(csv_files_dir))
        if id1 == 30:
            df1 = pd.read_csv(os.path.join(csv_files_dir, folders[id1], 'VIS_Prob_TEST.csv'))
            df1 = df1.fillna(0)
        else:
            df1 = pd.read_csv(os.path.join(csv_files_dir, folders[id1], 'VIS_Prob_TEST.csv'), index_col= 0)
            df1 = df1.fillna(0)

        if id2 == 30:
            df2 = pd.read_csv(os.path.join(csv_files_dir, folders[id2], 'VIS_Prob_TEST.csv'))
            df2 = df2.fillna(0)
        else:
            df2 = pd.read_csv(os.path.join(csv_files_dir, folders[id2], 'VIS_Prob_TEST.csv'), index_col= 0)
            df2 = df2.fillna(0)

        # Filter by indexes
        if idxs is not None:
            df1 = df1.iloc[idxs]
            df2 = df2.iloc[idxs]

        output = (df1.iloc[:, 0] - df2.iloc[:, 0])
        output = np.mean(output)
        return output


    def SHAP_Scores(idxs=None):
        Weights = [0.2, 0.2, 0.2, 0.2, 0.2,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.0333333,  0.0333333, 0.0333333, 0.0333333, 0.0333333, 0.0333333, 0.0333333, 0.0333333, 0.0333333, 0.0333333,
                   0.05, 0.05, 0.05, 0.05, 0.05,
                   0.2]

        G1_idx = [0, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 30]
        G1_weights = [Weights[index] for index in G1_idx]
        G1_Sub = [calc_mean_prob_df(0, 30, idxs=idxs),
                    calc_mean_prob_df(5, 1, idxs=idxs),
                    calc_mean_prob_df(6, 2, idxs=idxs),
                    calc_mean_prob_df(7, 3, idxs=idxs),
                    calc_mean_prob_df(8, 4, idxs=idxs),
                    calc_mean_prob_df(15, 9, idxs=idxs),
                    calc_mean_prob_df(16, 10, idxs=idxs),
                    calc_mean_prob_df(17, 11, idxs=idxs),
                    calc_mean_prob_df(18, 12, idxs=idxs),
                    calc_mean_prob_df(19, 13, idxs=idxs),
                    calc_mean_prob_df(20, 14, idxs=idxs),
                    calc_mean_prob_df(25, 21, idxs=idxs),
                    calc_mean_prob_df(26, 22, idxs=idxs),
                    calc_mean_prob_df(27, 23, idxs=idxs),
                    calc_mean_prob_df(28, 24, idxs=idxs),
                    calc_mean_prob_df(30, 29, idxs=idxs)]
        G1_score     = sum([a*b for a, b in zip(G1_weights, G1_Sub)])
        G1_score_abs = sum([a*abs(b) for a, b in zip(G1_weights, G1_Sub)])

        G2_idx = [1, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 25, 26, 27, 29, 30]
        G2_weights = [Weights[index] for index in G2_idx]
        G2_Sub = [calc_mean_prob_df(1, 30, idxs=idxs),
                  calc_mean_prob_df(5, 0, idxs=idxs),
                  calc_mean_prob_df(9, 2, idxs=idxs),
                  calc_mean_prob_df(10, 3, idxs=idxs),
                  calc_mean_prob_df(11, 4, idxs=idxs),
                  calc_mean_prob_df(15, 6, idxs=idxs),
                  calc_mean_prob_df(16, 7, idxs=idxs),
                  calc_mean_prob_df(17, 8, idxs=idxs),
                  calc_mean_prob_df(21, 12, idxs=idxs),
                  calc_mean_prob_df(22, 13, idxs=idxs),
                  calc_mean_prob_df(23, 14, idxs=idxs),
                  calc_mean_prob_df(25, 18, idxs=idxs),
                  calc_mean_prob_df(26, 19, idxs=idxs),
                  calc_mean_prob_df(27, 20, idxs=idxs),
                  calc_mean_prob_df(29, 24, idxs=idxs),
                  calc_mean_prob_df(30, 28, idxs=idxs)]
        G2_score     = sum([a*b for a, b in zip(G2_weights, G2_Sub)])
        G2_score_abs = sum([a*abs(b) for a, b in zip(G2_weights, G2_Sub)])

        G3_idx = [2, 6, 9, 12, 13, 15, 18, 19, 21, 22, 24, 25, 26, 28, 29, 30]
        G3_weights = [Weights[index] for index in G3_idx]
        G3_Sub = [calc_mean_prob_df(2, 30, idxs=idxs),
                  calc_mean_prob_df(6, 0, idxs=idxs),
                  calc_mean_prob_df(9, 1, idxs=idxs),
                  calc_mean_prob_df(12, 3, idxs=idxs),
                  calc_mean_prob_df(13, 4, idxs=idxs),
                  calc_mean_prob_df(15, 5, idxs=idxs),
                  calc_mean_prob_df(18, 7, idxs=idxs),
                  calc_mean_prob_df(19, 8, idxs=idxs),
                  calc_mean_prob_df(21, 10, idxs=idxs),
                  calc_mean_prob_df(22, 11, idxs=idxs),
                  calc_mean_prob_df(24, 14, idxs=idxs),
                  calc_mean_prob_df(25, 16, idxs=idxs),
                  calc_mean_prob_df(26, 17, idxs=idxs),
                  calc_mean_prob_df(28, 20, idxs=idxs),
                  calc_mean_prob_df(29, 23, idxs=idxs),
                  calc_mean_prob_df(30, 27, idxs=idxs)]
        G3_score = sum([a*b for a, b in zip(G3_weights, G3_Sub)])
        G3_score_abs = sum([a*abs(b) for a, b in zip(G3_weights, G3_Sub)])

        G4_idx = [3, 7, 10, 12, 14, 16, 18, 20, 21, 23, 24, 26, 27, 28, 29, 30]
        G4_weights = [Weights[index] for index in G4_idx]
        G4_Sub = [calc_mean_prob_df(3, 30, idxs=idxs),
                  calc_mean_prob_df(7, 0, idxs=idxs),
                  calc_mean_prob_df(10, 1, idxs=idxs),
                  calc_mean_prob_df(12, 2, idxs=idxs),
                  calc_mean_prob_df(14, 4, idxs=idxs),
                  calc_mean_prob_df(16, 5, idxs=idxs),
                  calc_mean_prob_df(18, 6, idxs=idxs),
                  calc_mean_prob_df(20, 8, idxs=idxs),
                  calc_mean_prob_df(21, 9, idxs=idxs),
                  calc_mean_prob_df(23, 11, idxs=idxs),
                  calc_mean_prob_df(24, 13, idxs=idxs),
                  calc_mean_prob_df(25, 15, idxs=idxs),
                  calc_mean_prob_df(27, 17, idxs=idxs),
                  calc_mean_prob_df(28, 19, idxs=idxs),
                  calc_mean_prob_df(29, 22, idxs=idxs),
                  calc_mean_prob_df(30, 26, idxs=idxs)]

        G4_score = sum([a*b for a, b in zip(G4_weights, G4_Sub)])
        G4_score_abs = sum([a*abs(b) for a, b in zip(G4_weights, G4_Sub)])

        G5_idx     = [4, 8, 11, 13, 14, 17, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30]
        G5_weights = [Weights[index] for index in G5_idx]
        G5_Sub     = [calc_mean_prob_df(4, 30, idxs=idxs),
                    calc_mean_prob_df(8, 0, idxs=idxs),
                    calc_mean_prob_df(11, 1, idxs=idxs),
                    calc_mean_prob_df(13, 2, idxs=idxs),
                    calc_mean_prob_df(14, 3, idxs=idxs),
                    calc_mean_prob_df(17, 5, idxs=idxs),
                    calc_mean_prob_df(19, 6, idxs=idxs),
                    calc_mean_prob_df(20, 7, idxs=idxs),
                    calc_mean_prob_df(22, 9, idxs=idxs),
                    calc_mean_prob_df(23, 10, idxs=idxs),
                    calc_mean_prob_df(24, 12, idxs=idxs),
                    calc_mean_prob_df(26, 15, idxs=idxs),
                    calc_mean_prob_df(27, 16, idxs=idxs),
                    calc_mean_prob_df(28, 18, idxs=idxs),
                    calc_mean_prob_df(29, 21, idxs=idxs),
                    calc_mean_prob_df(30, 25, idxs=idxs)]

        G5_score = sum([a*b for a, b in zip(G5_weights, G5_Sub)])
        G5_score_abs = sum([a*abs(b) for a, b in zip(G5_weights, G5_Sub)])

        scores = [G1_score,  G2_score, G3_score, G4_score, G5_score]
        scores_abs = [G1_score_abs,  G2_score_abs, G3_score_abs, G4_score_abs, G5_score_abs]

        return scores, scores_abs


    groups = ["G1", "G2", "G3", "G4", "G5"]
    dfSHAP = pd.DataFrame(groups, columns=["group"])

    score, score_abs = SHAP_Scores()
    dfSHAP["all"] = score
    dfSHAP["all_abs"] = score_abs

    if outcomes_file is not None:
        dfOutcomes = pd.read_csv(outcomes_file)
        outcome_types = dfOutcomes["outcome"].unique()

        for outcome_type in outcome_types:
            idxs = dfOutcomes[dfOutcomes["outcome"] == outcome_type].index.values
            score, score_abs = SHAP_Scores(idxs=idxs)
            dfSHAP["{}".format(outcome_type)] = score
            dfSHAP["{}_abs".format(outcome_type)] = score_abs

    dfSHAP.to_csv(output_file, index=None)

if __name__ == "__main__":
    main()
