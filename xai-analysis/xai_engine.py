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




def FIGW(model_object, input_data, input_target, xai_method = None, num_iter = None, random_state=42):
    # Feature Importnace Group-Wise (FIGW)

    # PFI: Permutation Feature Importance
    
    if  xai_method == 'PFI' or xai_method == 'GHO':
        Dict = utils.Permutation_Dict
        
    elif xai_method == 'LossSHAP':
        Dict = utils.LossSHAP_Dict
        
    df = pd.DataFrame(columns = ['Feature', 'PSS_mean', 'PSS_std', 'HSS_mean', 'HSS_std', 'CSS_mean', 'CSS_std'])
    this_hss, this_pss, this_css= [], [], []
    gnames = []    
        
        
    for i in num_iter: 

        for key in Dict:

            l = Dict[key]
            
            input_data_copy = copy.deepcopy(input_data)
            
            if xai_method == 'PFI': 
                for value in l:
                    
                    numpy.random.seed(random_state)
                    idx3 = numpy.random.permutation(input_data_copy[value].shape[1])
                    input_data_copy[value] = input_data_copy[value][:,idx3,:, :, :] 
                    
                
            elif xai_method == 'LossSHAP':
                
                for value in l:
                    numpy.random.seed(random_state)
                    input_data_copy[value]    = numpy.float32(numpy.random.randn(*input_data_copy[value].shape)) 


            y_testing_cat_prob = model_object.predict(input_data_copy) 
            metric_list        = cnn_evaluate.test_eval(input_target, y_testing_cat_prob, th = 0.193)
            
            this_pss.append(metric_list[8])
            this_hss.append(metric_list[9])
            this_css.append(metric_list[11])
            gnames.append(feature_name)
            

    df['group']    = gnames
    df['PSS_mean']   = this_pss
    df['PSS_std']    = numpy.std(this_pss)
    df['HSS_mean']   = this_hss
    df['HSS_std']    = numpy.std(this_hss)
    df['CSS_mean']   = this_css
    df['CSS_std']    = numpy.std(this_css)

    return df
#=========================================================================================================================================#
#=================================================== Permutation Feature Importance Channel-Wise =========================================#
#=========================================================================================================================================#
def PFI_channels(model_object, input_data, input_target, n_repeats=None, random_state=42): 

    df = pd.DataFrame(columns = ['Feature', 'PSS_mean', 'PSS_std', 'HSS_mean', 'HSS_std', 'CSS_mean', 'CSS_std'])
    #df = pd.DataFrame(columns = ['Feature', 'POD_mean', 'F_mean','FAR_mean', 'CSI_mean', 'PSS_mean', 'HSS_mean', 'ORSS_mean','CSS_mean'])
    #df = pd.DataFrame(columns = ['Feature', 'HSS_mean', 'HSS_std'])
    this_hss, this_pss, this_css= [], [], []
    fnames = []
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
            GNames = utils.NETCDF_PREDICTOR_NAMES['Mixed']
        elif g ==5:
            GNames = utils.NETCDF_PREDICTOR_NAMES['SST']        
        n_features = input_data[g].shape[3]

        for f in range(n_features):
            for i in range(n_repeats): 
                #print(f"We are in group {g}, with {n_features} features")

                input_data_copy = copy.deepcopy(input_data)

                numpy.random.seed(random_state)
                permuted_map = numpy.random.permutation(input_data_copy[g][:,:,:,f,:]) 
                input_data_copy[g][:,:,:,f,:] = permuted_map

                y_testing_cat_prob = model_object.predict(input_data_copy) 
                metric_list        = cnn_evaluate.test_eval(input_target, y_testing_cat_prob, th = 0.193)

                #this_pod.append(metric_list[0])
                #this_f.append(metric_list[1])
                #this_far.append(metric_list[2])
                #this_csi.append(metric_list[3])
                this_pss.append(metric_list[8])
                this_hss.append(metric_list[9])
                #this_orss.append(metric_list[6])
                this_css.append(metric_list[11])

                feature_name     = GNames[int(numpy.floor(f/4))]
                fnames.append(feature_name)

                #print(f"{feature_name}: HSS Mean = {this_hss}|") 


    #print(f"The calculation for feature {feature_name} is done!")
    df['Feature']    = fnames
    #df['POD_mean']   = this_pod
    #df['POD_std']    = numpy.std(this_pod)
    #df['F_mean']     = this_f
    #df['F_std']      = numpy.std(this_f)
   # df['FAR_mean']   = this_far
    #df['FAR_std']    = numpy.std(this_far)
    #df['CSI_mean']   = this_csi
    #df['CSI_std']    = numpy.std(this_csi)
    df['PSS_mean']   = this_pss
    df['PSS_std']    = numpy.std(this_pss)
    df['HSS_mean']   = this_hss
    df['HSS_std']    = numpy.std(this_hss)
    #df['ORSS_mean']  = this_orss
    #df['ORSS_std']   = numpy.std(this_orss)
    df['CSS_mean']   = this_css
    df['CSS_std']    = numpy.std(this_css)

    return df