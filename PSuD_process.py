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
import mcvqoe.math
import mcvqoe.simulation
import argparse

class PSuD_process():
    #TODO: Add documentation and docstrings to all methods    
    #TODO: Change to only take one path, assume has csv and wav in it
    def __init__(self,test_names, test_path='',fs = 48e3):
        self.test_names = test_names
        self.data_path = os.path.join(test_path,'csv')
        self.cp_path = os.path.join(test_path,'wav')
        self.fs = fs
        
        #TODO: Add audio length - store max we've seen, 
        self.max_audio_length = None
        
        if(type(self.test_names) is str):
            self.test_names = [self.test_names]
        
        self.test_dat, self.cps = self.load_sessions()
        
        #TODO: 
        
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
        
    def eval_psud(self,threshold,msg_len,p=0.95,R=1e4,uncertainty=True):
        
        # TODO: if msg_len > self.max_audio_length report NaN
        
        # Calculate test chains for this threshold, if we don't already have them
        if(threshold not in self.test_chains):
            self.get_test_chains(threshold)
        
        # Get relevant test chains
        test_chains = self.test_chains[threshold]
        
        # Label chains as success or failure
        msg_success = test_chains >= msg_len
        # Calculate fraction of tests that match msg_len requirement
        psud = np.mean(msg_success)
        
        if(uncertainty):
            # Calculate bootstrap uncertainty of msg success
            ci,_ = mcvqoe.math.bootstrap_ci(msg_success,p=p,R=R)
        else:
            ci = None
        
        
        return((psud,ci))
    

    
if(__name__ == "__main__"):
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description = __doc__)
    parser.add_argument('test_names',
                        type = str,
                        help = "Test names (same as name of folder for wav files)")
    parser.add_argument('-p', '--test-path',
                        default = '',
                        type = str,
                        help = "Path where test data is stored. Must contain wav and csv directories.")
    
    parser.add_argument('-f', '--fs',
                        default = 48e3,
                        type = int,
                        help = "Sampling rate for audio in tests")
    parser.add_argument('-t', '--threshold',
                        default = [],
                        nargs="+",
                        action = "extend",
                        type  = float,
                        help = "Intelligibility success threshold")
    parser.add_argument('-m', '--message-length',
                        default = [],
                        nargs = "+",
                        action = "extend",
                        type = float,
                        help = "Message length")
    
    args = parser.parse_args()
    

    t_proc = PSuD_process(args.test_names,
                          test_path= args.test_path,
                          fs = args.fs)
    # pdb.set_trace()
    for threshold in args.threshold:
        print("----Intelligibility Success threshold = {}----".format(threshold))
        print("Results shown as Psud(t) = mean, (95% C.I.)")
        msg_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
        for message_len in args.message_length:    
            psud_m,psud_ci = t_proc.eval_psud(threshold,message_len)
        
            print(msg_str.format(message_len,
                                  psud_m,
                                  psud_ci[0],
                                  psud_ci[1]
                                  ))
    # for thresh in [0.5, 0.7, 1]:
    #     print("----Intelligibility Success threshold = {}----".format(thresh))
    #     msg_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f}) | Expected = {:.4f} | Pass: {}"
    #     for msg_len in np.arange(1,11):
    #         psud_m,psud_ci = t_proc.eval_psud(thresh,msg_len)
    #         e_psud = pbi.expected_psud(msg_len)
    #         if(psud_ci[0] <= e_psud and e_psud <= psud_ci[1]):
    #             match = True
    #         else:
    #             match = False
            
    #         print(msg_str.format(msg_len,
    #                              psud_m,
    #                              psud_ci[0],
    #                              psud_ci[1],
    #                              e_psud,
    #                              match))
    