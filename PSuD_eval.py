#!/usr/bin/env python
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

class PSuD_eval():
    
    """
    Class to evaluate Probability of Successful Delivery tests
       
    Attributes
    ----------
    test_names : list
        Test file names to evaluate.
    
    test_info : dict
        Dict with data_path, data_file, and cp_path fields.
        
    fs : int
        Sample rate for audio
        
    test_dat : pd.DataFrame
        Data from all sessions
        
    cps : Dict
        Dict of each test, with cutpoints for each clip for each test.
        
    test_chains : np.array
        List of longest chain of successful words for each trial of a test
        
        
    max_audio_length : XXX
        Descr.
        
    Methods
    -------
    
    eval_psud()
        Determine the probability of successful delivery of a message
    
    """
    
    def __init__(self,test_names, test_path='',wav_dirs=[],fs = 48e3):
        """
        Initialize PSuD_eval object.

        Parameters
        ----------
        test_names : str or list
            Name of test, or list of names of tests.
        
        test_path : str
            Path where test data is stored. Does not need to be passed if 
            test_names contains full paths to files and wav_dirs is set as well.
            
        wav_dirs : str or list
            Paths to directories containing audio for a PSuD test. Must contain 
            cutpoints for audio clips in data files.
            
        fs : int
            Sample rate for audio. Default is 48e3 Hz

        Returns
        -------
        None.

        """
        if(isinstance(test_names,str)):
            test_names = [test_names]
            if(isinstance(wav_dirs,str)):
                wav_dirs=[wav_dirs]
        
        if(not wav_dirs):
            wav_dirs=(None,)*len(test_names)
        
        #initialize info arrays
        self.test_names = []
        self.test_info={}
        
        #loop through all the tests and find files
        for tn,wd in zip(test_names,wav_dirs):
            #split name to get path and name
            #if it's just a name all goes into name
            dat_path,name=os.path.split(tn)
            #split extension
            #again ext is empty if there is none
            #(unles there is a dot in the filename...) todo?
            t_name,ext=os.path.splitext(name)
            
            #check if a path was given to a .csv file
            if(not dat_path and not ext=='.csv'):
                #generate using test_path
                dat_path=os.path.join(test_path,'csv')
                dat_file=os.path.join(dat_path,t_name+'.csv')
                cp_path=os.path.join(test_path,'wav')
            else:
                cp_path=os.path.join(os.path.dirname(dat_path))
                dat_file=tn
            
            #check if we were given an explicit wav directory
            if(wd):
                #use given path
                cp_path=wd
                #get test name from wave path
                #normalize path first to remove a, possible, trailing slash
                t_name=os.path.basename(os.path.normpath(wd))
            else:
                #otherwise get path to the wav dir
                cp_path=os.path.join(cp_path,t_name)
                
            #put things into the test info structure
            self.test_info[t_name]={'data_path':dat_path,'data_file':dat_file,'cp_path':cp_path}
            #append name to list of names
            self.test_names.append(t_name)
        
        self.fs = fs
        
        #TODO: Add audio length - store max we've seen, 
        self.max_audio_length = None
        
        
        self.test_dat, self.cps = self.load_sessions()
        
        #TODO: 
        
        # Initialize empty dictionary to store test_chains in, keys will be threshold values
        self.test_chains = dict()
        
        
    def load_session(self,test_name):
        """
        Load a PSuD data session

        Parameters
        ----------
        test_name : str
            Name of PSuD test.

        Returns
        -------
        pd.DataFrame
            Session results
            
        dict
            Dictionary with cutpoints for each clip

        """
       
        fname = self.test_info[test_name]['data_file']

        test = pd.read_csv(fname)
        # Store test name as column in test
        test['name'] = test_name
        test_clips = self.get_clip_names(test)
        test_cp = {}
        for clip in test_clips:
            fpath = os.path.join(self.test_info[test_name]['cp_path'],"Tx_{}.csv".format(clip))
            test_cp[clip] = pd.read_csv(fpath)
        return((test,test_cp))
    
    def load_sessions(self):
        """
        Load and consolidate multiple PSuD Data Sessions for a given Test

        Returns
        -------
        pd.DataFrame
            Results from all sessions in self.test_names
            
        dict
            Dictionary of each test, with cutpoints for each clip for each test.

        """
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
        """
        Extract audio clip names from a session
        
        Parameters
        ----------
        test_dat : pd.DataFrame
            Data for a PSuD session
            
        Returns
        -------
        np.array
            Array of clip names
            
        """
        # Function to extract clip names from a session
        
        clip_names = np.unique(test_dat['Filename'])
        return(clip_names)
    
    def get_cutpoints(self,clip, test_name):
        """
        Load cutpoints for a given audio clip

        Parameters
        ----------
        clip : str
            Name of audio clip.
        test_name : str
            Name of test audio clip used in.

        Returns
        -------
        pd.DataFrame
            Cutpoints for clip

        """
        # Get cutpoints for clipname
        fpath = "{}.csv".format(os.path.join(self.cp_path,test_name,clip))
        cp = pd.read_csv(fpath)
        return(cp)
    
    def get_test_chains(self,threshold):
        """
        Determine longest successful chain of words for each trial of a test

        Parameters
        ----------
        threshold : float
            Intelligibility value used to determine whether a word was delivered successfully or not. Between [0,1]

        Returns
        -------
        np.array
            List of longest chain of successful words for each trial of a test

        """
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
        """
        Determine number of successfully deliverd words in a trial

        Parameters
        ----------
        trial : pd.Series
            Row of data from a session file.
        threshold : float
            Intelligibility value used to determine whether a word was delivered successfully or not. Between [0,1]

        Returns
        -------
        int
            Number of words that achieved intelligibility greater than threshold

        """
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
        """
        Convert word chain length to length in seconds

        Parameters
        ----------
        clip_cp : pd.DataFrame
            Cutpoints for the audio clip used for a given trial.
        chain_length : TYPE
            Number of consecutive words received successfully for a given trial.

        Returns
        -------
        float
            Time in seconds associated with last word or silent section in the audio clip before intelligibility threshold was not achieved.

        """
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
        
    def eval_psud(self,threshold,msg_len,p=0.95,R=1e4):
        """
        Determine the probability of successful delivery of a message

        Probability of successful delivery measures the probability of 
        successfully delivering a message of length msg_len. A message is 
        considered to have been successfully delivered it achieved an 
        intelligibility that is greater than threshold.
        
        Parameters
        ----------
        threshold : float
            Intelligibility value used to determine whether a word was delivered successfully or not. Between [0,1]
        msg_len : float
            Length of the message.
        p : float, optional
            Level of confidence to use in confidence interval calculation. Value must be in [0,1]. The default is 0.95.
        R : int, optional
            Number of repetitions used in uncertainy calculation using bootstrap resampling. The default is 1e4.

        Returns
        -------
        float
            probability of successful delivery. Between 0 and 1.
            
        np.array
            Confidence interval of level p for estimate.

        """
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
        
        if(len(msg_success) > 1):
            if(len(msg_success) < 30):
                warnings.warn("Number of samples is small. Reported confidence intervals may not be useful.")
            # Calculate bootstrap uncertainty of msg success
            ci,_ = mcvqoe.math.bootstrap_ci(msg_success,p=p,R=R)
        else:
            ci = np.array([-np.inf,np.inf])
        
        
        return((psud,ci))
    

    
if(__name__ == "__main__"):
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description = __doc__)
    parser.add_argument('test_names',
                        type = str,
                        nargs = "+",
                        action = "extend",
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
    

    t_proc = PSuD_eval(args.test_names,
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
    
