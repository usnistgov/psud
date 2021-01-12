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

# TODO: Consider designing a process class to simplify passing of paths and fs in these functions. Could just be class properties
class PSuD_process():
    
    def __init__(self,test_names, data_path='',cp_path='',fs = 48e3):
        self.test_names = test_names
        self.data_path = data_path
        self.cp_path = cp_path
        self.fs = fs
        
        if(type(self.test_names) is str):
            self.test_names = [self.test_names]
        
        self.test_dat, self.cps = self.load_sessions()
        
        # Initialize empty dictionary to store test_chains in, keys will be threshold values
        self.test_chains = dict()
        
        
    def load_session(self,test_name):
        fname = "{}.csv".format(os.path.join(self.data_path,test_name))
        test = pd.read_csv(fname)
        # Store test name as column in test
        test['name'] = test_name
        test_clips = self.get_clip_names(test)
        test_cp = {}
        for clip in test_clips:
            fpath = os.path.join(self.cp_path,test_name,"Tx_{}.csv".format(clip))
            test_cp[clip] = pd.read_csv(fpath)
        return((test,test_cp))
    
    def load_sessions(self):
        tests = pd.DataFrame()
        tests_cp = {}
        for test_name in self.test_names:
            test,test_cp = self.load_session(test_name)
            tests = tests.append(test)
            
            # Store cutoints as a dictionary of test names with each value being a dictionary of cutpoints for that test
            tests_cp[test_name] = test_cp
            
            # #NOTE: if tests_cp and test_cp share a dictionary key, the value is overwritten by the one in test_cp
            # if(update_warn == True):
            #     updates = set(tests_cp.keys()).intersection(set(test_cp.keys()))
            #     if(bool(updates)):
            #         warnings.warn("Cutpoints being overwritten: {}".format(updates))
                
            # tests_cp = {**tests_cp, **test_cp}
        return((tests,tests_cp))
    
    def get_clip_names(self,test_dat):
        # Function to extract clip names from a session
        
        clip_names = np.unique(test_dat['Filename'])
        return(clip_names)
    
    def get_cutpoints(self,clip, test_name):
        # Get cutpoints for clipname
        fpath = "{}.csv".format(os.path.join(self.cp_path,test_name,clip))
        cp = pd.read_csv(fpath)
        return(cp)
    
    def get_test_chains(self,threshold):
        chains = []
        times = []
        for ix,trial in self.test_dat.iterrows():
            chain_len = self.get_trial_chain_length(trial,threshold=threshold)
            chains.append(chain_len)
            
            # Get clip cutpoints
            clip_cp = self.cps[trial['name']][trial['Filename']]
            chain_time = self.chain2time(clip_cp,chain_len)
            times.append(chain_time)
        np_chains = np.array(chains)
        self.test_chains[threshold] = np_chains
        return(np_chains)
    
    def get_trial_chain_length(self,trial, threshold):
        
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
    
    def chain2time(self,clip_cp,chain_length):
        if(chain_length == 0):
            return(0)
        
        # Get indices for mrt keywords
        key_ix = np.where(~np.isnan(clip_cp['Clip']))[0]
        
        # Get index of last word in chain
        chain_ix = key_ix[chain_length-1]
        success_ix = chain_ix
        silence_flag = True
        
        while(silence_flag):
            if(success_ix < (len(clip_cp)-1)):
                if(np.isnan(clip_cp.loc[success_ix+1,'Clip'])):
                    # If next word is silence, increment success index 
                    #Note: we deem silence to be assumed to still be success
                    success_ix += 1
                else:
                    silence_flag = False
            else:
                # Have reached last word, so success index is last chunk of message
                silence_flag = False
        
        # chain time: increment end sample by 1 to make numbers round better
        chain_time = (clip_cp.loc[success_ix,'End']+1)/self.fs
        
        return(chain_time)
        
    def eval_psud(self,threshold,msg_len):
        # Calculate test chains for this threshold, if we don't already have them
        if(threshold in self.test_chains):
            self.get_test_chains(threshold)
        
        # Get relevant test chains
        test_chains = self.test_chains[threshold]
        # Calculate fraction of tests that match msg_len requirement
        psud = sum(test_chains >= msg_len)/len(test_chains)
        return(psud)
    
    
        

# def main(filenames,data_path,cp_path,threshold,fs):
#     # Load test data
#     test_dat,cps = load_sessions(filenames,
#                              data_path=data_dir,
#                              cp_path=cp_path,
#                              update_warn = False)
    
    
#     # test_cp = []
#     # for clip in test_clips:
#     #     test_cp.append(get_cutpoints(clip, cp_path))
    
#     test_chains = get_test_chains(test_dat,
#                                   cps,
#                                   threshold,
#                                   fs)
    
#     for msg_len in np.arange(1,11):
#         psud_m = eval_psud(test_chains,msg_len)
#         print("PSuD({}) = {}".format(msg_len,psud_m))
if(__name__ == "__main__"):
    
    data_dir = os.path.join("data","csv")
    cp_dir = os.path.join("data","wav")
    # Data generated from:
    # python PSuD_simulate.py --audioPath D:\MCV_671DRDOG\Audio-Clips\PSuD_Clip_10  --audioFiles F1_PSuD_Norm_10.wav F3_PSuD_Norm_10.wav M3_PSuD_Norm_10.wav M4_PSuD_Norm_10.wav --trials 100 --P-a1 0.95 --P-a2 0.95 --P-r 0.95 --P-interval 0.2 -P
    
    thresh = 1
    
    filenames = ["capture_simulation_test_07-Jan-2021_09-43-49.csv",
                 "capture_simulation_test_08-Jan-2021_15-29-48.csv"]
    tests = ["capture_simulation_test_07-Jan-2021_09-43-49",
                 "capture_simulation_test_08-Jan-2021_15-29-48"]
    fs = 48e3
    
    t_proc = PSuD_process(tests, 
                          data_path=data_dir,
                          cp_path=cp_dir,
                          fs = 48e3)
    
    tchains = t_proc.get_test_chains(thresh)
    for msg_len in np.arange(1,11):
        psud_m = t_proc.eval_psud(thresh,msg_len)
        print("PSuD({}) = {}".format(msg_len,psud_m))
    