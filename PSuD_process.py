# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:37:32 2021

@author: jkp4
"""
import numpy as np
import pandas as pd
import os
import warnings
import pdb

def load_session(test_name,data_path,cp_path):
    fname = "{}.csv".format(os.path.join(data_path,test_name))
    test = pd.read_csv(fname)
    test_clips = get_clip_names(test)
    test_cp = {}
    for clip in test_clips:
        fpath = os.path.join(cp_path,test_name,"Tx_{}.csv".format(clip))
        test_cp[clip] = pd.read_csv(fpath)
    return((test,test_cp))

def load_sessions(test_names,data_path,cp_path,update_warn = False):
    tests = pd.DataFrame()
    tests_cp = {}
    for test_name in test_names:
        test,test_cp = load_session(test_name,data_path,cp_path)
        tests = tests.append(test)
        #NOTE: if tests_cp and test_cp share a dictionary key, the value is overwritten by the one in test_cp
        if(update_warn == True):
            updates = set(tests_cp.keys()).intersection(set(test_cp.keys()))
            if(bool(updates)):
                warnings.warn("Cutpoints being overwritten: {}".format(updates))
            
        tests_cp = {**tests_cp, **test_cp}
    return((tests,test_cp))

def get_clip_names(test_dat):
    # Function to extract clip names from a session
    
    clip_names = np.unique(test_dat['Filename'])
    return(clip_names)

def get_cutpoints(clip, test_name, cp_path):
    # Get cutpoints for clipname
    fpath = "{}.csv".format(os.path.join(cp_path,test_name,clip))
    cp = pd.read_csv(fpath)
    return(cp)

def get_test_chains(test_dat,cps,threshold,fs):
    chains = []
    times = []
    for ix,trial in test_dat.iterrows():
        chain_len = get_trial_chain_length(trial,threshold=threshold)
        chains.append(chain_len)
        
        chain_time = chain2time(cps[trial['Filename']],chain_len,fs)
        times.append()
    return(chains)

def get_trial_chain_length(trial, threshold):
    
    # flag to determine if chain is still active
    chain = True
    # Length of current chain
    chain_len = 0
    
    while(chain):
        # Convert chain length to word
        colname = "W{}_Int".format(chain_len)
        if colname in trial and trial[colname] >= threshold:
            #Check that both colname exists in trial and that trial success 
            #is greater than threshold
            
            # Increment chain length
            chain_len += 1
        else:
            # Break chain
            chain = False
    return(chain_len)

def chain2time(clip_cp,chain_length,fs,high_res = False):
    print("this function converts chain length to time")
    pdb.set_trace()
    # Extract only keyword cutpoints (ignore NaN)
    keywords = clip_cp.loc[~np.isnan(clip_cp['Clip'])]
    
    keywords.index = np.arange(len(keywords))
    # TODO: Make this smarter and determine high_res mode/how to calculate end time by looking ahead from chain_length to wherever there is the next entry that is not nan...
    if(high_res):
        # High res mode assumes that there is no silence between keywords
        chain_time = keywords.loc[chain_length,'End']/fs
    else:
        
        not_keywords = clip_cp.loc[np.isnan(clip_cp['Clip'])]
        not_keywords.index = np.arange(len(not_keywords))
        chain_time = not_keywords.loc[chain_length,'End']/fs
    return(chain_time)
    
    
def main(filenames,data_path,cp_path,threshold,fs):
    # Load test data
    test_dat,cps = load_sessions(filenames,
                             data_path=data_dir,
                             cp_path=cp_path,
                             update_warn = False)
    
    
    
    # test_cp = []
    # for clip in test_clips:
    #     test_cp.append(get_cutpoints(clip, cp_path))
    
    test_chains = get_test_chains(test_dat,
                                  cps,
                                  threshold,
                                  fs)
    pdb.set_trace()
if(__name__ == "__main__"):
    
    data_dir = os.path.join("data","csv")
    cp_dir = os.path.join("data","wav")
    # Data generated from:
    # python PSuD_simulate.py --audioPath D:\MCV_671DRDOG\Audio-Clips\PSuD_Clip_10  --audioFiles F1_PSuD_Norm_10.wav F3_PSuD_Norm_10.wav M3_PSuD_Norm_10.wav M4_PSuD_Norm_10.wav --trials 100 --P-a1 0.95 --P-a2 0.95 --P-r 0.95 --P-interval 0.2 -P
    # fname = "capture_simulation_test_07-Jan-2021_09-43-49.csv"
    
    # fpath = os.path.join(data_dir,fname)
    
    thresh = 1
    
    filenames = ["capture_simulation_test_07-Jan-2021_09-43-49.csv",
                 "capture_simulation_test_08-Jan-2021_15-29-48.csv"]
    tests = ["capture_simulation_test_07-Jan-2021_09-43-49",
                 "capture_simulation_test_08-Jan-2021_15-29-48"]
    fs = 48e3
    main(tests,
         data_path = data_dir,
         cp_path = cp_dir,
         threshold = thresh,
         fs = fs)
    
    
    