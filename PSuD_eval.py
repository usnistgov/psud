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

class evaluate():
    
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
    
    def __init__(self,test_names, test_path='',wav_dirs=[],fs = 48e3,use_reprocess = True):
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
                
                # remove possible R in t_name
                wt_name = t_name.replace('Rcapture','capture')
                cp_path=os.path.join(cp_path,wt_name)
                
            #put things into the test info structure
            self.test_info[t_name]={'data_path':dat_path,'data_file':dat_file,'cp_path':cp_path}
            #append name to list of names
            self.test_names.append(t_name)
        
        self.fs = fs
        
        #TODO: Add audio length - store max we've seen, 
        self.max_audio_length = None
        
        # Check if you reprocessed data should be used
        self.use_reprocess = use_reprocess
        
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
        
        
        if(self.use_reprocess):
            # Look for reprocessed file if it exists
            fname = self.check_reprocess(fname)
        
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
        # Ensure that tests has unique row index
        nrow,_ = tests.shape
        tests.index = np.arange(nrow)
        return((tests,tests_cp))
    
    def check_reprocess(self,fname):
        """
        Look for a reprocessed data file in same path as fname
        
        Searches for a reprocessed data file in same path as fname. 
        Reprocessed data always starts as 'Rcapture', where original data 
        starts with 'capture'. Returns reprocessed file name if it exists, 
        otherwise returns original file name.

        Parameters
        ----------
        fname : str
            Path to a session csv file.

        Returns
        -------
        str:
            Path to reprocessed file if it exits, otherwise returns fname

        """
        dat_path,name=os.path.split(fname)
        if('Rcapture' not in name):
            reprocess_fname = os.path.join(dat_path, 'R{}'.format(name))
            if(os.path.exists(reprocess_fname)):
                out_name = reprocess_fname
            else:
                out_name = fname
        else:
            out_name = fname
        
        return(out_name)
    
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
    
    def get_test_chains(self,method,threshold,method_weight = None):
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
        
        if(method =="ARF"):
            method_intell = self.filter_intelligibility(self.test_dat.filter(regex="W\d+_Int"),
                                                     weight=method_weight)
        elif(method == "AMI"):
            if(method_weight is not None):
                warnings.warn("Method weight passed to AMI, will not be used")
            method_intell = self.smooth_intelligibility(self.test_dat.filter(regex="W\d+_Int"))
        elif(method == "EWC"):
            if(method_weight is not None):
                warnings.warn("Method weight passed to EWC, will not be used")
            method_intell = self.test_dat.filter(regex="W\d+_Int")
        else:
            raise ValueError('Invalid method passed: {}'.format(method))
        
        for ix,trial in self.test_dat.iterrows():
            chain_len = self.get_trial_chain_length(method_intell.loc[ix],threshold)
            # if(method == "EWC"):
            #     chain_len = self.get_trial_chain_length(trial.filter(regex='W\d+_Int'),threshold=threshold)
            # elif(method == "ARF"):
            #     chain_len = self.get_trial_chain_length(arf_intell.loc[ix],threshold=threshold)
                # chain_len = None
            chains.append(chain_len)
            
            # Get clip cutpoints
            clip_cp = self.cps[trial['name']][trial['Filename']]
            chain_time = self.chain2time(clip_cp,chain_len)
            times.append(chain_time)
        np_chains = np.array(chains)
        if(method in self.test_chains):
            self.test_chains[method][threshold] = np_chains
        else:
            self.test_chains[method] = {threshold: np_chains}
                
        # self.test_chains[threshold] = np_chains
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
             
        failures = np.where(~(trial >= threshold))
        if(failures[0].size == 0):
            chain_len = len(trial)
        else:
            chain_len = failures[0][0]
        return(chain_len)
        # # Length of current chain
        # chain_len = 0
        
        # for word_int in trial:t
        #     if(word_int)
        
        # while(chain):
        #     # Convert chain length to word
        #     colname = "W{}_Int".format(chain_len)
        #     if colname in trial and trial[colname] >= threshold:
        #         #Check that both colname exists in trial and that trial success 
        #         #is greater than threshold
                
        #         # Increment chain length
        #         chain_len += 1
        #     else:
        #         # Break chain
        #         chain = False
        # return(chain_len)
    
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
    
    def filter_intelligibility(self,int_data,weight=0.5):
        
        # # Get number ofrows in data
        # nrow,_ = int_data.shape
        # # Ensure that int_data has unique row names
        # int_data.index = np.arange(nrow)
        
        # Initialize new data frame
        fint = pd.DataFrame(columns = int_data.columns)
        
        for ix,trial in  int_data.iterrows():
            # initialize array to store filtered trial data in
            ftrial = np.empty(len(trial))
            for wix, wint in enumerate(trial):
                if(wix == 0):
                    # For first word, intelligibility is just first word intelligibility
                    ftrial[wix] = trial[wix]
                else:
                    ftrial[wix] = weight*trial[wix] + (1-weight)*ftrial[wix-1]
            
            # Store new trial intelligibility
            fint.loc[ix] = ftrial
        return(fint)
            
    def smooth_intelligibility(self,int_data):
        
        # Initialize new data frame
        fint = pd.DataFrame(columns = int_data.columns)
        for ix,trial in  int_data.iterrows():
            # initialize array to store filtered trial data in
            ftrial = np.empty(len(trial))
            for wix, wint in enumerate(trial):
                # Average intelligibility up to this word
                ftrial[wix] = np.mean(trial[:(wix+1)])
            # Store new trial intelligibility    
            fint.loc[ix] = ftrial
        return(fint)
    
    def eval_psud(self,threshold,msg_len,p=0.95,R=1e4,method='EWC',method_weight = None):
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
        if(method not in self.test_chains):
            self.get_test_chains(method,threshold,method_weight=method_weight)
        elif(threshold not in self.test_chains[method]):
            self.get_test_chains(method,threshold,method_weight=method_weight)
        # if(threshold not in self.test_chains):
        #     self.get_test_chains(threshold)
        
        # Get relevant test chains
        test_chains = self.test_chains[method][threshold]
        
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
    
    def clear(self):
        self.test_chains = dict()
    

#--------------------------------[main]---------------------------------------
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
    
    parser.add_argument('-n', '--no-reprocess',
                        default = True,
                        action = "store_false",
                        help = "Do not use reprocessed data if it exists.")
    parser.add_argument('--method',
                        default = "EWC",
                        type = str,
                        help = "PSuD method to use. Must be one of 'EWC' or 'ARF'.")
    parser.add_argument('-w','--method-weight',
                        default = 0.5,
                        type = float,
                        help = 'Weight for method filters if applicable.')
    
    args = parser.parse_args()
    
    t_proc = evaluate(args.test_names,
                          test_path= args.test_path,
                          fs = args.fs,
                          use_reprocess=args.no_reprocess)
    
    
    
    for threshold in args.threshold:
        print("----Intelligibility Success threshold = {}----".format(threshold))
        print("Results shown as Psud(t) = mean, (95% C.I.)")
        msg_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
        for message_len in args.message_length:    
            psud_m,psud_ci = t_proc.eval_psud(threshold,
                                              message_len,
                                              method = args.method,
                                              method_weight = args.method_weight)
            
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
    
# Test Analog direct:
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'Rcapture_Analog-direct_11-Feb-2021_14-23-10 Rcapture_Analog-direct_11-Feb-2021_11-45-21 Rcapture_Analog-direct_11-Feb-2021_09-51-59 Rcapture_Analog-direct_11-Feb-2021_06-22-06 -p data -m 1 3 5 10 -t 0.5 0.7 0.9 --method ARF')

# Test P25 Direct
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'capture_P25-direct_22-Feb-2021_14-18-29 capture_P25-direct_22-Feb-2021_12-24-20 capture_P25-direct_22-Feb-2021_10-02-11 capture_P25-direct_23-Feb-2021_07-01-38 -p data -m 1 3 5 10 -t 0.5 0.7 0.9 --method ARF')

# Test P25 Trunked Phase 1
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'capture_P25-Trunked-p1_23-Feb-2021_11-28-27 capture_P25-Trunked-p1_23-Feb-2021_09-03-52 capture_P25-Trunked-p1_24-Feb-2021_07-16-10 capture_P25-Trunked-p1_23-Feb-2021_13-24-39 -p data -t 0.5 0.7 0.9 -m 1 3 5 10 --method ARF')

# Test P25 Trunked Phase 2
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'capture_P25-P2_25-Feb-2021_07-54-07 capture_P25-P2_25-Feb-2021_09-52-10 capture_P25-P2_24-Feb-2021_09-25-15 capture_P25-P2_24-Feb-2021_12-16-08 -p data -t 0.5 0.7 0.9 -m 1 3 5 10 --method ARF')