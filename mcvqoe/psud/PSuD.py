import abcmrt
import csv
import glob
import mcvqoe.base
import mcvqoe.delay
import pkg_resources
import re
import shutil
import time
import sys
import os.path
import datetime

import numpy as np

from distutils.util import strtobool
from mcvqoe.base.terminal_user import terminal_progress_update
# version import for logging purposes
from .version import version


def chans_to_string(chans):
    """
    Convert list of audio channels to string.

    Parameters
    ----------
    chans : list
        List of audio channels in a measurement.

    Returns
    -------
    str
        Semicolon separated string of channels.

    """
    #channel string
    return '('+(';'.join(chans))+')'


def parse_audio_chans(csv_str):
    '''
    Function to parse audio channels from csv file
    '''
    match=re.search('\((?P<chans>[^)]+)\)',csv_str)

    if(not match):
        raise ValueError(f'Unable to parse chans {csv_str}, expected in the form "(chan1;chan2;...)"')

    return tuple(match.group('chans').split(';'))


class measure(mcvqoe.base.Measure):
    """
    Class to run and reprocess Probability of Successful Delivery tests.

    The PSuD class is used to run Probability of Successful Delivery tests.
    These can either be tests with real communication devices or simulated Push
    To Talk (PTT) systems.
    
    Attributes
    ----------
    audio_files : list
        List of names of audio files. relative paths are relative to audio_path
    audio_path : string
        Path where audio is stored
    overPlay : float
        Number of extra seconds of audio to record at the end of a trial
    trials : int
        Number of times audio will be run through the system in the run method
    outdir : string
        Base directory where data is stored.
    ri : mcvqoe.RadioInterface or mcvqoe.QoEsim
        Object to use to key the audio channel
    info : dict
        Dictionary with test info to for the log entry
    ptt_wait : float
        Time to wait, in seconds, between keying the channel and playing audio
    ptt_gap : float
        Time to pause, in seconds, between one trial and the next
    rng : Generator
        Generator to use for random numbers
    audio_interface : mcvqoe.AudioPlayer or mcvqoe.simulation.QoEsim
        interface to use to play and record audio on the communication channel
    time_expand : 1 or 2 element list or tuple of floats
        Amount of time, in seconds, of extra audio to use for intelligibility
        estimation. If only one value is given, it is used both before and after
        the clip
    m2e_min_corr : float
        minimum correlation to accept for a good mouth to ear measurement.
        Values range from 1 (perfect correlation) to 0 (no correlation)
    get_post_notes : function or None
        Function to call to get notes at the end of the test. Often set to
        mcvqoe.post_test to get notes with a gui popup.
        lambda : mcvqoe.post_test(error_only=True) can be used if notes should
        only be gathered when there is an error
    intell_est : {'trial','post','none'}
        String to control when intelligibility and mouth to ear estimations are
        done. Should behavior is as follows:
        'trial' to estimate them after each trial is complete
        'post' will estimate them after all trials have finished,
        'none' will not compute intelligibility or M2E at all and will store
            dummy values in the .csv file.
        Any other value is treated the same as 'none'
    split_audio_dest : string or None
        if this is a string it holds the path where individually cut word clips
        are stored. this directory will be created if it does not exist
    data_fields : dict
        static property that has info on the standard .csv columns. Column names
        are dictionary keys and the values are conversion functions to get from
        string to the appropriate type. This should not be modified in most
        cases
    no_log : tuple of strings
        static property that is a tuple of property names that will not be added
        to the 'Arguments' field in the log. This should not be modified in most
        cases
    y : list of audio vectors
        Audio data for transmit clips. This is set by the load_audio function.
    cutpoints : list of lists of dicts
        list of cutpoints for corresponding transmit clips. This is set by the
        load_audio function.
    keyword_spacings : list of floats
        time, in seconds, of the most closely spaced words in a clip. This is
        set by the load_audio function.
    time_expand_samples : two element list of ints
        time expand values in samples. This is automatically generated from
        time_expand in `run` and `post_process`. These values are used
        internally to time expand the cutpoints
    num_keywords : int
        the maximum number of keywords in a single audio clip. This is used when
        making the .csv as it dictates how many columns the .csv has for word
        intelligibility. This is set automatically in the audio_clip_check
        method and should not normally need to be set
    clipi : list of ints
        list containing the indices of the transmit clip that is used for each
        trial. This is randomized in `run` before the test is run
    data_filename : string
        This is set in the `run` method to the path to the output .csv file.
    full_audio_dir : bool, default=False
        read, and use, .wav files in audio_path, ignore audio_files and trials
    progress_update : function, default=terminal_progress_update
        function to call to provide updates on test progress. This function
        takes three arguments, progress type, total number of trials, current
        trial number. The first argument, progress type is a string that will be
        one of {'test','proc'} to indicate that the test is running trials or
        processing data.
    save_tx_audio : bool, default=True
        Save transmitted audio in output directory. Overridden by save_audio
    save_audio : bool, default=True
        Save transmitted and received audio

    Methods
    -------

    run()
        run a test with the properties of the class
    load_test_data(fname,load_audio=True)
        load dat from a .csv file. If load_audio is true then the Tx clips from
        the wav dir is loaded into the class. returns the .csv data as a list of
        dicts
    post_process(test_dat,fname,audio_path)
        process data from load_test_dat and write a new .csv file.

    Examples
    --------
    example of running a test with simulated devices.

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
    >>> test_obj.run()
    
    Example of reprocessing  a test file, 'test.csv', to get 'rproc.csv'
    # TODO: Fix this example!
    
    >>>from PSuD_1way_1loc import PSuD
    >>>test_obj=PSuD()
    >>>test_dat=test_obj.load_test_data('[path/to/outdir/]data/csv/test.csv')
    >>>test_obj.post_process(test_dat,'rproc.csv',test_obj.audio_path)
    """

    measurement_name = "PSuD"

    #on load conversion to datetime object fails for some reason
    #TODO : figure out how to fix this, string works for now but this should work too:
    #row[k]=datetime.datetime.strptime(row[k],'%d-%b-%Y_%H-%M-%S')
    data_fields={"Timestamp":str,"Filename":str,"m2e_latency":float,"good_M2E":(lambda s: bool(strtobool(s))),"channels":parse_audio_chans,"Over_runs":int,"Under_runs":int}
    no_log=('rng','y','clipi','data_dir','wav_data_dir','csv_data_dir','cutpoints','data_fields','time_expand_samples','num_keywords')
    
    def __init__(self, **kwargs):
        """
        create a new PSuD object.

        """
                 
        self.rng=np.random.default_rng()
        #set default values
        self.trials = 100
        self.outdir = ''
        self.ri = None
        self.info = {'Test Type':'default','Pre Test Notes':''}
        self.ptt_wait = 0.68
        self.ptt_gap = 3.1
        self.audio_interface = None
        self.time_expand = [100e-3 - 0.11e-3, 0.11e-3]
        self.m2e_min_corr = 0.76
        self.get_post_notes = None
        self.test = "1loc"
        self.split_audio_dest = None
        self.progress_update = terminal_progress_update
        self.save_tx_audio = True
        self.save_audio = True
        self.p_thresh = -np.inf

        # Get all included audio file sets
        self._default_audio_sets, self._default_audio_path = self.included_audio_sets()
        
        self.audio_files = []
        # Make default audio first audio set
        self.audio_path = os.path.join(self._default_audio_path,
                                       self._default_audio_sets[0])
        self.full_audio_dir = True

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError(f"{k} is not a valid keyword argument")

    @staticmethod
    def included_audio_sets():
        """
        Return audio sets and paths included in the package.
    
        Returns
        -------
        audio_sets : list
            List of all included audio sets.
        audio_path : str
            Path to audio sets.
    
        """
        audio_sets = pkg_resources.resource_listdir(
            'mcvqoe.psud', 'audio_clips'
            )
        # Get path to audio file sets
        audio_path = pkg_resources.resource_filename(
            'mcvqoe.psud', 'audio_clips'
            )
        return audio_sets, audio_path

    def load_audio(self):
        """
        load audio files for use in test.
        
        this loads audio from self.audio_files and stores values in self.y,
        self.cutpoints and self.keyword_spacings
        In most cases run() will call this automatically but, it can be called
        in the case that self.audio_files is changed after run() is called

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError
            If self.audio_files is empty
        RuntimeError
            If clip fs is not 48 kHz
        """
   
        #if we are not using all files, check that audio files is not empty
        if not self.audio_files and not self.full_audio_dir:
            #TODO : is this the right error to use here??
            raise ValueError('Expected self.audio_files to not be empty')

        #check if we are making split audio
        if(self.split_audio_dest):
            #make sure that splid audio directory exists
            os.makedirs(self.split_audio_dest,exist_ok=True)
            
        if(self.full_audio_dir):
            #override audio_files
            self.audio_files=[]
            #look through all things in audio_path
            for f in os.scandir(self.audio_path):
                #make sure this is a file
                if(f.is_file()): 
                    #get extension
                    _,ext=os.path.splitext(f.name)
                    #check for .wav files
                    if(ext=='.wav'):
                        #add to list
                        self.audio_files.append(f.name)
                #TODO : recursive search?

        #list for input speech
        self.y=[]
        #list for cutpoints
        self.cutpoints=[]
        #list for word spacing
        self.keyword_spacings=[]
        
        for f in self.audio_files:
            #make full path from relative paths
            f_full=os.path.join(self.audio_path,f)
            # load audio
            fs_file, audio_dat = mcvqoe.base.audio_read(f_full)
            #check fs
            if(fs_file != abcmrt.fs):
                raise RuntimeError(f'Expected fs to be {abcmrt.fs} but got {fs_file} for {f}')
            # Convert to float sound array and add to list
            self.y.append(audio_dat)
            #strip extension from file
            fne,_=os.path.splitext(f_full)
            #add .csv extension
            fcsv=fne+'.csv'
            #load cutpoints
            cp=mcvqoe.base.load_cp(fcsv)
            #add cutpoints to array
            self.cutpoints.append(cp)
            
            starts=[]
            ends=[]
            lens=[]
            for cpw in cp:
                if(np.isnan(cpw['Clip'])):
                    #check if this is the first clip, if so skip
                    #TODO: deal with this better?
                    if(ends):
                        ends[-1]=cpw['End']
                        lens[-1]=ends[-1]-starts[-1]
                else:
                    starts.append(cpw['Start'])
                    ends.append(cpw['End'])
                    lens.append(ends[-1]-starts[-1])
            
            #word spacing is minimum distance converted to seconds
            self.keyword_spacings.append(min(lens)/abcmrt.fs)
            
    def set_time_expand(self,t_ex):
        """
        convert time expand from seconds to samples and ensure a 2 element vector.
        
        This is called automatically in run and post_process and, normally, it
        is not required to call set_time_expand manually

        Parameters
        ----------
        t_ex :
            time expand values in seconds
        Returns
        -------
        """
        self.time_expand_samples=np.array(t_ex)
        
        if(len(self.time_expand_samples)==1):
            #make symmetric interval
            self.time_expand_samples=np.array([self.time_expand_samples,]*2)

        #convert to samples
        self.time_expand_samples=np.ceil(
                self.time_expand_samples*abcmrt.fs
                ).astype(int)
        
    def audio_clip_check(self):
    #TODO : this could probably be moved into load_audio, also not 100% sure this name makes sense
        """
        find the number of keywords in clips.
        
        this is called when loading audio in `run` and load_test_dat it should
        not, normally, need to be called manually

        Parameters
        ----------
        
        Returns
        -------
        """
        #number of keyword columns to have in the .csv file
        self.num_keywords=0
        #check cutpoints and count keywaords
        for cp in self.cutpoints:
            #count the number of actual keywords
            n=sum(not np.isnan(w['Clip']) for w in cp)
            #set num_keywords to max values
            self.num_keywords=max(n,self.num_keywords)
            
        if(self.full_audio_dir):
            #overide trials to use all the trials
            self.trials=len(self.y)


    def csv_header_fmt(self):
        """
        generate header and format for .csv files.
        
        This generates a header for .csv files along with a format (that can be
        used with str.format()) to generate each row in the .csv
        
        Parameters
        ----------
        
        Returns
        -------
        hdr : string
            csv header string
        fmt : string
            format string for data lines for the .csv file
        """
        hdr=','.join(self.data_fields.keys())
        fmt='{'+'},{'.join(self.data_fields.keys())+'}'
        for word in range(self.num_keywords):
            hdr+=f',W{word}_Int'
            fmt+=f',{{intel[{word}]}}'
        #add newlines at the end
        hdr+='\n'
        fmt+='\n'
        
        return (hdr,fmt)
        
    def log_extra(self):
        #add abcmrt version
        self.info['abcmrt version']=abcmrt.version
        
    def test_setup(self):
        #-----------------------[Check audio sample rate]-----------------------
        if self.audio_interface is not None and \
            self.audio_interface.sample_rate != abcmrt.fs:
            raise ValueError(f'audio_interface sample rate is {self.audio_interface.sample_rate} Hz but only {abcmrt.fs} Hz is supported')
        #---------------------------[Set time expand]---------------------------
        self.set_time_expand(self.time_expand)
        
    def process_audio(self,clip_index,fname,rec_chans):
        """
        estimate mouth to ear latency and intelligibility for an audio clip.

        Parameters
        ----------
        clip_index : int
            index of the matching transmit clip. can be found with find_clip_index
        fname : str
            audio file to process

        Returns
        -------
        dict
            returns a dictionary with estimated values

        """
        
        #---------------------[Load in recorded audio]---------------------
        fs, rec_dat = mcvqoe.base.audio_read(fname)
        if(abcmrt.fs != fs):
            raise RuntimeError('Recorded sample rate does not match!')
        
        #check if we have more than one channel
        if(rec_dat.ndim !=1 ):
            #get the index of the voice channel
            voice_idx=rec_chans.index('rx_voice')
            #get voice channel
            voice_dat=rec_dat[:,voice_idx]
        else:
            voice_dat=rec_dat

        rec_dat = voice_dat
        
        #------------------------[calculate M2E]------------------------
        pos,dly = mcvqoe.delay.ITS_delay_est(self.y[clip_index], voice_dat, "f", fs=abcmrt.fs,min_corr=self.m2e_min_corr)
        
        if(not pos):
            #M2E estimation did not go super well, try again but restrict M2E bounds to keyword spacing
            pos,dly = mcvqoe.delay.ITS_delay_est(self.y[clip_index], voice_dat, "f", fs=abcmrt.fs,dlyBounds=(0,self.keyword_spacings[clip_index]))
            
            good_m2e=False
        else:
            good_m2e=True
             
        estimated_m2e_latency=dly / abcmrt.fs

        #---------------------[Compute intelligibility]---------------------
        
        #strip filename for basename in case of split clips
        if(isinstance(self.split_audio_dest, str)):
            (bname,_)=os.path.splitext(os.path.basename(fname))
        else:
            bname=None

        success=self.compute_intelligibility(
                                            voice_dat,
                                            self.cutpoints[clip_index],
                                            dly,
                                            clip_base=bname
                                            )

            
        return {
                    'm2e_latency':estimated_m2e_latency,
                    'intel':success,
                    'good_M2E':good_m2e,
                    'channels':chans_to_string(rec_chans),
                }

    def compute_intelligibility(self,audio,cutpoints,cp_shift,clip_base=None):
        """
        estimate intelligibility for audio.

        Parameters
        ----------
        audio : audio vector
            time aligned audio to estimate intelligibility on
        cutpoints : list of dicts
            cutpoints for audio file
        cp_shift : int
            Offset to add to cutpoints to correct for M2E.
        clip_base : str or None, default=None
            basename for split clips. Split clips will not be written if None

        Returns
        -------
        numpy.array
            returns a vector of intelligibility values padded to self.num_keywords

        """
        #----------------[Cut audio and perform time expand]----------------

        #array of audio data for each word
        word_audio=[]
        #array of word numbers
        word_num=[]
        #maximum index
        max_idx=len(audio)-1
        
        for cp_num,cpw in enumerate(cutpoints):
            if(not np.isnan(cpw['Clip'])):
                #calculate start and end points
                start=np.clip(cp_shift+cpw['Start']-self.time_expand_samples[0],0,max_idx)
                end  =np.clip(cp_shift+cpw['End']  +self.time_expand_samples[1],0,max_idx)
                #add word audio to array
                word_audio.append(audio[start:end])
                #add word num to array
                word_num.append(cpw['Clip'])

                if(clip_base and isinstance(self.split_audio_dest, str)):
                    outname=os.path.join(self.split_audio_dest,f'{clip_base}_cp{cp_num}_w{cpw["Clip"]}.wav')
                    #write out audio
                    mcvqoe.base.audio_write(outname, int(abcmrt.fs), audio[start:end])

        #---------------------[Compute intelligibility]---------------------
        phi_hat,success=abcmrt.process(word_audio,word_num)
        
        if not np.isneginf(self.p_thresh):
            for n, wa in enumerate(word_audio):
                #do a-weight checks
                pw = mcvqoe.base.a_weighted_power(wa,abcmrt.fs)
                if pw < self.p_thresh:
                    #power low, zero
                    success[n] = 0

        #expand success so len is num_keywords
        success_pad=np.empty(self.num_keywords)
        success_pad.fill(np.nan)
        success_pad[:success.shape[0]]=success
        
        return success_pad
