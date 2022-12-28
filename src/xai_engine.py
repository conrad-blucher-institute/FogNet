import pandas as pd
import numpy
import os
import os.path 
import copy
import errno
import glob
import time
import calendar
import json
import pickle
import random
import netCDF4
import statistics
from src import utils, FogNet, FogNetConfig, cnn_evaluate


def FIGW(model_object, input_data, input_target, xai_method = None, num_iter = None, random_state=42):
    # Feature Importnace Group-Wise (FIGW)

    # PFI: Permutation Feature Importance
    
    if  xai_method == 'PFI' or xai_method == 'GHO':
        Dict = utils.Permutation_Dict
        
    elif xai_method == 'LossSHAP':
        Dict = utils.LossSHAP_Dict
        
    df = pd.DataFrame()
    
    mean_hss, mean_pss, mean_css = [], [], []
    std_hss, std_pss, std_css = [], [], []
    gnames = []    
    
     
        
    for key in Dict:
        
        this_hss, this_pss, this_css= [], [], []
        for i in range(num_iter):

            l = Dict[key]
            
            input_data_copy = copy.deepcopy(input_data)
            
            if xai_method == 'PFI': 
                for value in l:
                    
                    numpy.random.seed(random_state)
                    idx3                   = numpy.random.permutation(input_data_copy[value].shape[1])
                    input_data_copy[value] = input_data_copy[value][:,idx3,:, :, :] 
                
            elif xai_method == 'LossSHAP':
                
                for value in l:
                    #numpy.random.seed(random_state)
                    input_data_copy[value] = numpy.float32(numpy.random.randn(*input_data_copy[value].shape)) 
                

            y_testing_cat_prob = model_object.predict(input_data_copy, batch_size = 1) 
            metric_list        = cnn_evaluate.test_eval(input_target, y_testing_cat_prob, threshold = 0.193)
            print(f"{metric_list[0]}|{metric_list[1]}|{metric_list[2]}|{metric_list[3]}")
            
            this_pss.append(metric_list[8])
            this_hss.append(metric_list[9])
            this_css.append(metric_list[11])
            
            print(f"{xai_method} in iteration {i}")
            print(f"{key}| PSS = {metric_list[8]}| HSS = {metric_list[9]}| CSS = {metric_list[11]}")

            
        mean_pss.append(statistics.mean(this_pss))
        std_pss.append(statistics.pstdev(this_pss))

        mean_hss.append(statistics.mean(this_hss))
        std_hss.append(statistics.pstdev(this_hss))

        mean_css.append(statistics.mean(this_css))
        std_css.append(statistics.pstdev(this_css)) 

        gnames.append(key)
    
    
    df['group']    = gnames
    df['PSS_Mean'] = mean_pss
    df['PSS_STD']  = std_pss
    
    df['HSS_Mean'] = mean_hss
    df['HSS_STD']  = std_hss
    
    df['CSS_Mean'] = mean_css
    df['CSS_STD']  = std_css
    
    return df
#=========================================================================================================================================#
#=================================================== Permutation Feature Importance Channel-Wise =========================================#
#=========================================================================================================================================#
def PFI_channels(model_object, input_data, input_target, n_repeats=None, random_state=42): 

    df = pd.DataFrame()
    
    mean_hss, mean_pss, mean_css = [], [], []
    std_hss, std_pss, std_css = [], [], []
    fnames, gnames = [], []
    
    n_groups   = len(input_data)
    
    for g in range(n_groups): 
        
        if g ==0:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G1']
        elif g ==1:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G2']
        elif g ==2:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G3']
        elif g ==3:
            GNames = utils.NETCDF_PREDICTOR_NAMES['Physical_G4']
        elif g ==4:
            GNames = utils.NETCDF_MIXED_NAMES
        elif g ==5:
            GNames = utils.NETCDF_MUR_NAMES        
        n_features = input_data[g].shape[3]
        
        
        for f in range(n_features):
            
            this_hss, this_pss, this_css = [], [], []

            for i in range(n_repeats):
                #print(f"Iteration {i}: group {g}, feature name {GNames[f]}!")
                
                input_data_copy    = copy.deepcopy(input_data)
                numpy.random.seed(random_state)
                permuted_map       = numpy.random.permutation(input_data_copy[g][:,:,:,f,:]) 
                input_data_copy[g][:,:,:,f,:] = permuted_map

                y_testing_cat_prob = model_object.predict(input_data_copy, batch_size = 4) 
                metric_list        = cnn_evaluate.test_eval(input_target, y_testing_cat_prob, threshold = 0.193)

                
                this_pss.append(metric_list[8])
                this_hss.append(metric_list[9])
                this_css.append(metric_list[11])

            feature_name   = GNames[f]
            fnames.append(feature_name)
            gnames.append(g)
            
            mean_pss.append(statistics.mean(this_pss))
            std_pss.append(statistics.pstdev(this_pss))

            mean_hss.append(statistics.mean(this_hss))
            std_hss.append(statistics.pstdev(this_hss))

            mean_css.append(statistics.mean(this_css))
            std_css.append(statistics.pstdev(this_css)) 

            print(f"{GNames[f]}| PSS = {statistics.mean(this_pss):.2f}, {statistics.pstdev(this_pss):.2f}| HSS = {statistics.mean(this_hss):.2f}, {statistics.pstdev(this_hss):.2f}| CSS = {statistics.mean(this_css):.2f}, {statistics.pstdev(this_css):.2f}")


    df['Group']      = gnames
    df['Feature']    = fnames
    df['PSS_mean']   = mean_pss
    df['PSS_std']    = std_pss
    df['HSS_mean']   = mean_hss
    df['HSS_std']    = std_hss
    df['CSS_mean']   = mean_css
    df['CSS_std']    = std_css

    return df