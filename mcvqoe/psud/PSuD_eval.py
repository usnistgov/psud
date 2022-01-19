#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:37:32 2021

@author: jkp4
"""
import argparse
import itertools
import json
import os
import re
import warnings


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mcvqoe.math
import mcvqoe.simulation

# TODO: Deprecate ARF and clean up overly general method things. Just make clean split between AMI vs EWC to make code easier to follow
class evaluate():
    """
    Class to evaluate Probability of Successful Delivery tests.

    Class to evaluate PSuD tests. Processes either one or a group of tests.
    When passed a list of test files all the data is aggregated and PSuD is
    determined from the results from all tests.

    Parameters
    ----------
    test_names : str or list
        Name of test, or list of names of tests.

    test_path : str
        Path where test data is stored. Does not need to be passed if
        test_names contains full paths to files and wav_dirs is set as
        well.

    wav_dirs : str or list
        Paths to directories containing audio for a PSuD test. Must contain
        cutpoints for audio clips in data files.

    fs : int
        Sample rate for audio. Default is 48e3 Hz.

    use_reprocess : bool
        Use reprocessed data rather than original data if it exists for this
        test. Default is True.

    Attributes
    ----------
    test_names : list
        Test file names to evaluate.

    test_info : dict
        Dict with data_path, data_file, and cp_path fields.

    fs : int
        Sample rate for audio

    data : pd.DataFrame
        Data from all sessions

    cps : Dict
        Dict of each test, with cutpoints for each clip for each test.

    test_chains : np.array
        List of longest chain of successful words for each trial of a test

    max_audio_length : XXX
        Descr.

    Methods
    -------
    eval()
        Determine the probability of successful delivery of a message

    See Also
    --------
    mcvqoe.psud.measure : Measurement class for generating PSuD data.

    Examples
    --------
    Run a simulation and evaluate EWC PSuD for an intelligibility threshold of
    0.5 and a message of length 3.

    >>> import mcvqoe.simulation
    >>> sim_obj=mcvqoe.simulation.QoEsim()
    >>> test_obj=mcvqoe.psud.measure(ri=sim_obj,audio_interface=sim_obj,
    ...                              trials=10,audio_path='path/to/audio/',
    ...                              audio_files=('F1_PSuD_Norm_10.wav',
    ...                                           'F3_PSuD_Norm_10.wav',
    ...                                           'M3_PSuD_Norm_10.wav',
    ...                                           'M4_PSuD_Norm_10.wav'
    ...                                          )
    ...                             )
    >>> fname = test_obj.run()
    >>> psud_proc = mcvqoe.psud.evaluate(fname)
    >>> ewc_psud = psud_proc.eval(0.5,3)

    Using same evaluation object evaluate AMI PSuD for an intelligibility
    threshold of 0.7 and a message of length 10
    >>> ami_psud = psud_proc.eval(0.7,10)
    """

    def __init__(self,
                 test_names=None,
                 test_path='',
                 wav_dirs=[],
                 use_reprocess=True,
                 json_data=None,
                 **kwargs):
        self.use_reprocess = use_reprocess
        if json_data is not None:
            self.test_names, self.test_info, self.data, self.cps = self.load_json_data(json_data)
        else:
            if isinstance(test_names, str):
                test_names = [test_names]
                if isinstance(wav_dirs, str):
                    wav_dirs = [wav_dirs]
    
            if not wav_dirs:
                wav_dirs = (None, ) * len(test_names)
    
            # initialize info arrays
            self.test_names = []
            self.test_info = {}
    
            # loop through all the tests and find files
            for tn, wd in zip(test_names, wav_dirs):
                # split name to get path and name
                # if it's just a name all goes into name
                dat_path, name = os.path.split(tn)
                # split extension
                # again ext is empty if there is none
                # (unles there is a dot in the filename...) todo?
                t_name, ext = os.path.splitext(name)
    
                # check if a path was given to a .csv file
                if not dat_path and not ext == '.csv':
                    # generate using test_path
                    dat_path = os.path.join(test_path, 'csv')
                    dat_file = os.path.join(dat_path, t_name+'.csv')
                    cp_path = os.path.join(test_path, 'wav')
                else:
                    cp_path = os.path.join(os.path.dirname(dat_path), 'wav')
                    dat_file = tn
    
                # check if we were given an explicit wav directory
                if wd:
                    # use given path
                    cp_path = wd
                    # get test name from wave path
                    # normalize path first to remove a, possible, trailing slash
                    t_name = os.path.basename(os.path.normpath(wd))
                else:
                    # otherwise get path to the wav dir
    
                    # remove possible R in t_name
                    wt_name = t_name.replace('Rcapture', 'capture')
                    cp_path = os.path.join(cp_path, wt_name)
    
                # put things into the test info structure
                self.test_info[t_name] = {'data_path': dat_path,
                                          'data_file': dat_file,
                                          'cp_path': cp_path}
                # append name to list of names
                self.test_names.append(t_name)
                
                self.data, self.cps = self.load_sessions()

        # TODO: Add audio length - store max we've seen,
        self.max_audio_length = None

        # Initialize empty dictionary to store test_chains in, keys will be
        # threshold values
        self.test_chains = dict()

        self.fs = 48e3
        self.message_averages = None
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError(f"{k} is not a valid keyword argument")
    
    def __repr__(self):
        call_trace = f'{type(self).__module__}.{type(self).__name__}'
        if self.test_info == 'json':
            out = call_trace + f' loaded from json, original test_names={self.test_names}'
        else:
            tpaths = []
            cppaths = []
            for tname, tinfo in self.test_info.items():
                tpaths.append(tinfo['data_file'])
                cppaths.append(tinfo['cp_path'])
            out = call_trace + '(' + f'test_names={tpaths}, ' + f'wav_dirs={cppaths}' + ')'
         
        return out
    
    
    def load_session(self, test_name):
        """
        Load a PSuD data session.

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

        if self.use_reprocess:
            # Look for reprocessed file if it exists
            fname = self.check_reprocess(fname)

        test = pd.read_csv(fname)
        # Store test name as column in test
        test['name'] = test_name
        test_clips = self.get_clip_names(test)
        test_cp = {}
        for clip in test_clips:
            fpath = os.path.join(self.test_info[test_name]['cp_path'],
                                 "Tx_{}.csv".format(clip))
            test_cp[clip] = pd.read_csv(fpath)
        return (test, test_cp)

    def load_sessions(self):
        """
        Load and consolidate multiple PSuD Data Sessions for a given Test.

        Returns
        -------
        pd.DataFrame
            Results from all sessions in self.test_names

        dict
            Dictionary of each test, with cutpoints for each clip for each
            test.

        """
        tests = pd.DataFrame()
        tests_cp = {}
        for test_name in self.test_names:
            test, test_cp = self.load_session(test_name)
            tests = tests.append(test)

            # Store cutoints as a dictionary of test names with each value
            # being a dictionary of cutpoints for that test
            tests_cp[test_name] = test_cp

        # Ensure that tests has unique row index
        nrow, _ = tests.shape
        tests.index = np.arange(nrow)
        return (tests, tests_cp)
    

    def check_reprocess(self, fname):
        """
        Look for a reprocessed data file in same path as fname.

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
        dat_path, name = os.path.split(fname)
        if 'Rcapture' not in name:
            reprocess_fname = os.path.join(dat_path, 'R{}'.format(name))
            if os.path.exists(reprocess_fname):
                out_name = reprocess_fname
            else:
                out_name = fname
        else:
            out_name = fname

        return out_name

    def get_clip_names(self, test_dat):
        """
        Extract audio clip names from a session.

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
        return clip_names
    
    def to_json(self, filename=None):
        """
        Create json representation of psud data

        Parameters
        ----------
        filename : str, optional
            If given save to json file. Otherwise returns json string. The default is None.

        Returns
        -------
        None.

        """
        
        cps = {}
        for sesh, sesh_cps in self.cps.items():
            cps[sesh] = {}
            for talker_word, cp in sesh_cps.items():
                cps[sesh][talker_word] = cp.to_json()
        
        out_json = {
            'measurement': self.data.to_json(),
            'cps': cps,
            'test_info': self.test_info,
            # 'test_info': json.dumps(self.test_info),
            
                }
        
        # Final json representation of all data
        final_json = json.dumps(out_json)
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(out_json, f)
        
        return final_json
    
    def load_json_data(self, json_data):
        """
        Do all data loading from input json_data

        Parameters
        ----------
        json_data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        # Extract data, cps, and test_info from json_data
        data = pd.read_json(json_data['measurement'])
        cps = {}
        cp_data = json_data['cps']
        for sesh, sesh_cps in cp_data.items():
            cps[sesh] = {}
            for talker_word, cp in sesh_cps.items():
                cps[sesh][talker_word] = pd.read_json(cp)
        
        test_info = json_data['test_info']
        
        # Return normal Access data attributes from these
        return test_info.keys(), test_info, data, cps
        
    
    @property
    def max_message_length(self):
        """
        Determine maximum message length in seconds from cutpoints

        Returns
        -------
        max_word : float
            Maximum message length in seconds.

        """
        max_audio_length = -np.Inf
        for session, cps in self.cps.items():
            for clip, cp in cps.items():
                audio_len = np.max(cp['End'])/self.fs
                max_audio_length = np.max([max_audio_length, audio_len])
        
        max_audio_length += 1/self.fs
        return max_audio_length
        
    def get_test_chains(self, method, threshold, method_weight=None):
        """
        Determine longest successful chain of words for each trial of a test.

        Parameters
        ----------
        method : str
            PSuD method to determine message success. One of EWC
            (every word critical), AMI (average message intelligibility), or
            ARF (autoregressive filter). ARF relies on an additional
            method_weight parameter as it is of the form
            method_weight*(int of word k) + (1-method_weight)*(int of word k-1)

        method_weight : float
            Weighting parameter to control impact of a single word for a
            method's revised intelligibility. Generally structured so that
            method_weight*word_int + (1-method_weight)*separate_factor.

        threshold : float
            Intelligibility value used to determine whether a word was
            delivered successfully or not. Between [0,1]

        Returns
        -------
        np.array
            List of longest chain of successful words for each trial of a test

        """
        chains = []
        times = []

        if method == "ARF":
            method_intell = self.ARF_intelligibility(
                self.data.filter(regex=r"W\d+_Int"),
                weight=method_weight)
        elif method == "AMI":
            if method_weight is not None:
                warnings.warn("Method weight passed to AMI, will not be used")
            method_intell = self.AMI_intelligibility(
                self.data.filter(regex=r"W\d+_Int"))
        elif method == "EWC":
            if method_weight is not None:
                warnings.warn("Method weight passed to EWC, will not be used")
            method_intell = self.data.filter(regex=r"W\d+_Int")
        else:
            raise ValueError('Invalid method passed: {}'.format(method))

        if method == "AMI":
            # Test chains don't matter for AMI
            np_chains = method_intell
        else:
            for ix, trial in self.data.iterrows():

                chain_len = self.get_trial_chain_length(method_intell.loc[ix],
                                                        threshold)
                chains.append(chain_len)

                # Get clip cutpoints
                clip_cp = self.cps[trial['name']][trial['Filename']]
                chain_time = self.chain2time(clip_cp, chain_len)
                times.append(chain_time)
            np_chains = np.array(chains)
        if method in self.test_chains:
            self.test_chains[method][threshold] = np_chains
        else:
            self.test_chains[method] = {threshold: np_chains}

        # self.test_chains[threshold] = np_chains
        return np_chains

    def get_trial_chain_length(self, trial, threshold):
        """
        Determine number of successfully deliverd words in a trial.

        Parameters
        ----------
        trial : pd.Series
            Row of data from a session file.
        threshold : float
            Intelligibility value used to determine whether a word was
            delivered successfully or not. Between [0,1]

        Returns
        -------
        int
            Number of words that achieved intelligibility greater than
            threshold

        """
        failures = np.where(~(trial >= threshold))
        if failures[0].size == 0:
            chain_len = len(trial)
        else:
            chain_len = failures[0][0]
        return chain_len

    def chain2time(self, clip_cp, chain_length):
        """
        Convert word chain length to length in seconds.

        Parameters
        ----------
        clip_cp : pd.DataFrame
            Cutpoints for the audio clip used for a given trial.
        chain_length : TYPE
            Number of consecutive words received successfully for a given
            trial.

        Returns
        -------
        float
            Time in seconds associated with last word or silent section in the
            audio clip before intelligibility threshold was not achieved.

        """
        if chain_length == 0:
            return 0

        # Get indices for mrt keywords
        key_ix = np.where(~np.isnan(clip_cp['Clip']))[0]

        # Get index of last word in chain
        chain_ix = key_ix[chain_length-1]
        success_ix = chain_ix
        silence_flag = True

        while silence_flag:
            if success_ix < (len(clip_cp)-1):
                if np.isnan(clip_cp.loc[success_ix+1, 'Clip']):
                    # If next word is silence, increment success index
                    # Note: we deem silence to be assumed to still be success
                    success_ix += 1
                else:
                    silence_flag = False
            else:
                # Have reached last word, so success index is last chunk of
                # message
                silence_flag = False

        # chain time: increment end sample by 1 to make numbers round better
        chain_time = (clip_cp.loc[success_ix, 'End']+1)/self.fs

        return chain_time

    def ARF_intelligibility(self, int_data, weight=0.5):
        """
        Autoregressive filter on intelligibility.

        Filter intelligibility by trial through an autoregressive filter (ARF),
        ARF relies on an additional method_weight parameter as it is of the
        form
        method_weight*(int of word k) + (1-method_weight)*(int of word k-1).
        If int_data is nxm, the results are an nxm dataframe. The first column
        of the output will always be identical to the first column of the
        input.

        Parameters
        ----------
        int_data : pd.DataFrame
            Data frame with each row representing a trial and each column
            containing intelligibility data for a subsequent word in a trial.

        weight : float, optional
            Weight of a given word in the filter. The default is 0.5.

        Returns
        -------
        pd.DataFrame
            Filtered intelligibility results.

        """
        # # Get number ofrows in data
        # nrow,_ = int_data.shape
        # # Ensure that int_data has unique row names
        # int_data.index = np.arange(nrow)

        # Initialize new data frame
        fint = pd.DataFrame(columns=int_data.columns)

        for ix, trial in int_data.iterrows():
            # initialize array to store filtered trial data in
            ftrial = np.empty(len(trial))
            for wix, wint in enumerate(trial):
                if wix == 0:
                    # For first word, intelligibility is just first word
                    # intelligibility
                    ftrial[wix] = trial[wix]
                else:
                    ftrial[wix] = weight*trial[wix] + (1-weight)*ftrial[wix-1]

            # Store new trial intelligibility
            fint.loc[ix] = ftrial
        return fint

    def AMI_intelligibility(self, int_data):
        """
        Average message intelligibility up to each word in a message.

        Smooth intelligibility by calculating average message intelligibility
        up to each word in the message.

        Parameters
        ----------
        int_data : pd.DataFrame
            Data frame with each row representing a trial and each column
            containing intelligibility data for a subsequent word in a trial.

        Returns
        -------
        pd.DataFrame
            Smoothed intelligibility results.

        """
        # Initialize new data frame
        fint = pd.DataFrame(columns=int_data.columns)
        for ix, trial in int_data.iterrows():
            # initialize array to store filtered trial data in
            ftrial = np.empty(len(trial))
            for wix, wint in enumerate(trial):
                # Average intelligibility up to this word
                ftrial[wix] = np.mean(trial[:(wix+1)])
            # Store new trial intelligibility
            fint.loc[ix] = ftrial
        return fint

    def eval(self,
             threshold,
             msg_len,
             p=0.95,
             R=1e4,
             method='EWC',
             method_weight=None):
        """
        Determine the probability of successful delivery of a message.

        Probability of successful delivery measures the probability of
        successfully delivering a message of length msg_len. A message is
        considered to have been successfully delivered it achieved an
        intelligibility that is greater than threshold.

        Parameters
        ----------
        threshold : float
            Intelligibility value used to determine whether a word was
            delivered successfully or not. Between [0,1].

        msg_len : float
            Length of the message.

        p : float, optional
            Level of confidence to use in confidence interval calculation.
            Value must be in [0,1]. The default is 0.95.

        R : int, optional
            Number of repetitions used in uncertainy calculation using
            bootstrap resampling. The default is 1e4.

        method : str
            PSuD method to determine message success. One of EWC
            (every word critical), AMI (average message intelligibility), or
            ARF (autoregressive filter). ARF relies on an additional
            method_weight parameter as it is of the form
            method_weight*(int of word k) + (1-method_weight)*(int of word k-1)

        method_weight : float
            Weighting parameter to control impact of a single word for a
            method's revised intelligibility. Generally structured so that
            method_weight*word_int + (1-method_weight)*separate_factor.

        Returns
        -------
        float
            probability of successful delivery. Between 0 and 1.

        np.array
            Confidence interval of level p for estimate.

        """
        # TODO: if msg_len > self.max_audio_length report NaN
        # TODO: Be smarter about AMI...test chains don't cahge
        # Calculate test chains for this threshold, if we don't already have
        # them
        if method == 'AMI':
            
            if self.message_averages is None:
                self.message_averages = self.AMI_intelligibility(
                    self.data.filter(regex=r"W\d+_Int")
                    )
            msg_success = []
            for ix, trial in self.message_averages.iterrows():
                check_ix = np.floor(msg_len - 1).astype(int)
                msg_success.append(trial[check_ix] >= threshold)
        else:
            if method not in self.test_chains:
                self.get_test_chains(method,
                                     threshold,
                                     method_weight=method_weight)
            
            elif threshold not in self.test_chains[method]:
                self.get_test_chains(method,
                                     threshold,
                                     method_weight=method_weight)
            # if threshold not in self.test_chains:
            #     self.get_test_chains(threshold)
    
            # Get relevant test chains
            test_chains = self.test_chains[method][threshold]
            
            # Label chains as success or failure
            msg_success = test_chains >= msg_len

        # if method == "AMI":
        #     msg_success = []
        #     for ix, trial in test_chains.iterrows():
        #         check_ix = np.floor(msg_len - 1).astype(int)
        #         msg_success.append(trial[check_ix] >= threshold)

        # else:
        #     # Label chains as success or failure
        #     msg_success = test_chains >= msg_len

        # Calculate fraction of tests that match msg_len requirement
        psud = np.mean(msg_success)

        if len(msg_success) > 1:
            if len(msg_success) < 30:
                warnings.warn(("Number of samples is small."
                               "Reported confidence intervals may not be"
                               "useful."))
            # Calculate bootstrap uncertainty of msg success
            ci, _ = mcvqoe.math.bootstrap_ci(msg_success, p=p, R=R)
        else:
            ci = np.array([-np.inf, np.inf])

        return (psud, ci)

    def clear(self):
        """
        Clear contents of test_chains.

        Returns
        -------
        None.

        """
        self.test_chains = dict()
        
    def plot(self, methods, thresholds,
             title='Probability of Successful Delivery by message length'):
        # methods = ['EWC', 'AMI']
        # thresholds = [0.5, 0.7]
        message_lengths = np.arange(self.max_message_length + 1)
        results = []
        for method, thresh, msg_len, in itertools.product(methods, thresholds, message_lengths):
            val, ci = self.eval(thresh, msg_len, method=method)
            res = {
                'Method': method,
                'Intelligibility Threshold': thresh,
                'Message Length (s)': msg_len,
                'PSuD': val,
                'Confidence Lower Bound': ci[0],
                'Confidence Upper Bound': ci[1],
                }
            results.append(res)
        df = pd.DataFrame(results)
        # fig = px.scatter(df,
        #                  x='Message Length (s)',
        #                  y='PSuD',
        #                  color='Method',
        #                  symbol='Intelligibility Threshold',
        #                  )
        fig = px.line(df,
                      x='Message Length (s)',
                      y='PSuD',
                      color='Method',
                      symbol='Intelligibility Threshold',
                      title=title,
                      )
        return fig
        
        
    def plot_intelligibility(self, data='message', test_name=None, x=None,
                             talkers=None,
                             title='Intelligibility Scatter Plot'):
        """
        Plot PSuD intelligibility data, either message or word. Message
        averages intelligibility over each word in the trial.

        Parameters
        ----------
        data : str, optional
            Type of intelligibility to plot. Either 'message' or 'word'.
            The default is 'message'.

        Returns
        -------
        None.

        """
        columns = self.data.columns
        
        # Find all intelligibility columns
        intell_cols = []
        other_cols = []
        word_search = re.compile('W(\d+)_Int')
        for col in columns:
            ws = word_search.search(col)
            if ws is not None:
                intell_cols.append(col)
            else:
                other_cols.append(col)
        if data == 'message':
            df = self.data
            df['Intell'] = df[intell_cols].mean(axis=1)
        elif data == 'word':
            df = pd.melt(self.data,
                         id_vars=other_cols,
                         value_vars=intell_cols,
                         value_name='Intell',
                         )
            df = df.sort_values(by=['Timestamp', 'variable'])
            df.index = np.arange(len(df))
        else:
            raise ValueError(f'Invalid data given \'{data}\'. Must be either \'message\' or \'word\'.')
        
        # Filter by session name if given
        if test_name is not None:
            df_filt = pd.DataFrame()
            if not isinstance(test_name, list):
                test_name = [test_name]
            for name in test_name:
                df_filt = df_filt.append(df[df['name'] == name])
            df = df_filt
       # Filter by talkers if given
        if talkers is not None:
            df_filt = pd.DataFrame()
            if isinstance(talkers, str):
                talkers = [talkers]
            for talker in talkers:
                ix = [talker in x for x in df['Filename']]
                df_sub = df[ix]
                df_sub['Talker'] = talker
                df_filt = df_filt.append(df_sub)
                
            df = df_filt
        else:
            # TODO: Consider just dropping this into init/data load, might make things easier
            pattern = re.compile(r'([FM]\d)(?:_n\d+_s\d+_c\d+)')
            talkers = set()
            talker_v = []
            for index, row in df.iterrows():
                res = pattern.search(row['Filename'])
                if res is not None:
                    talker = res.groups()[0]
                    talkers.add(talker)
                    talker_v.append(talker)
                else:
                    talker_v.append('NA')
            df['Talker'] = talker_v 
        # Set x-axis value
        if x is None:
            x = df.index
        if data=='message':
            symbol = 'Talker'
        else:
            symbol = 'variable'
        fig = px.scatter(df, x=x, y='Intell',
                          color='name',
                          symbol=symbol,
                          hover_name='Filename',
                          title=title,
                          labels={
                              'index': 'Trial Number',
                              'Intell': 'Intelligibility',
                              },
                          )
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.99,
            xanchor="left",
            x=0.01,
            ),
            legend_orientation="h",
            showlegend=False,
        )
        return fig
    
    def histogram(self,test_name=None, talkers=None,
                  title='Histogram of longest EWC Messages'):
        # fig = go.Figure()
        # Make sure something exists in test chains
        if 'EWC' not in self.test_chains:
            _ = self.eval(0.5, 3, method='EWC')
        df = pd.DataFrame()
        for thresh, chains in self.test_chains['EWC'].items():
            df_tmp = pd.DataFrame()
            df_tmp['Chain'] = chains
            df_tmp['Threshold'] = thresh
            df = df.append(df_tmp)
            # fig.add_trace(
            #     go.Histogram(
            #         x=chains,
            #         # color_discrete_sequence=thresh,
            #         )
            #     )
        fig = px.histogram(df,
                           x='Chain',
                           color='Threshold',
                           title=title,
                           labels={
                               'Chain': 'Consecutive Successful Words',
                               },
                           )
        return fig

def main():
    """
    Evalaute PSuD data with command line arguments.

    Returns
    -------
    None.

    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('test_names',
                        type=str,
                        nargs="+",
                        action="extend",
                        help=("Test names (same as name of folder for wav"
                              "files)"))
    parser.add_argument('-p', '--test-path',
                        default='',
                        type=str,
                        help=("Path where test data is stored. Must contain"
                              "wav and csv directories."))

    parser.add_argument('-f', '--fs',
                        default=48e3,
                        type=int,
                        help="Sampling rate for audio in tests")
    parser.add_argument('-t', '--threshold',
                        default=[],
                        nargs="+",
                        action="extend",
                        type=float,
                        help="Intelligibility success threshold")
    parser.add_argument('-m', '--message-length',
                        default=[],
                        nargs="+",
                        action="extend",
                        type=float,
                        help="Message length")

    parser.add_argument('-n', '--no-reprocess',
                        default=True,
                        action="store_false",
                        help="Do not use reprocessed data if it exists.")
    parser.add_argument('--method',
                        default="EWC",
                        type=str,
                        help=("PSuD method to use. Must be one of 'EWC' or"
                              "'ARF'."))
    parser.add_argument('-w', '--method-weight',
                        default=0.5,
                        type=float,
                        help='Weight for method filters if applicable.')

    args = parser.parse_args()

    t_proc = evaluate(args.test_names,
                      test_path=args.test_path,
                      fs=args.fs,
                      use_reprocess=args.no_reprocess)

    for threshold in args.threshold:
        print(
            "----Intelligibility Success threshold = {}----".format(threshold))
        print("Results shown as Psud(t) = mean, (95% C.I.)")
        msg_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
        for message_len in args.message_length:
            psud_m, psud_ci = t_proc.eval(threshold,
                                               message_len,
                                               method=args.method,
                                               method_weight=args.method_weight)

            print(msg_str.format(message_len,
                                 psud_m,
                                 psud_ci[0],
                                 psud_ci[1]))

# --------------------------------[main]---------------------------------------


if __name__ == "__main__":
    main()
