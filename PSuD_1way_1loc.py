#!/usr/bin/env python

import argparse
import os.path
import scipy.io.wavfile
import numpy as np
import datetime
import shutil
import time
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav
import warnings
import csv
from distutils.util import strtobool

import mcvqoe
from abcmrt import ABC_MRT16


#offset rx audio so that M2E latency is removed
#TODO : maybe this should be in a comon library?
def align_audio(tx,rx,m2e_latency,fs):
    # create time array for the rx_signal and offest it by the calculated delay
    points = np.arange(len(rx)) / fs - m2e_latency

    # create interpolation function based on offsetted time array
    f = interp1d(points, rx, fill_value=np.nan)
    
    try:
        # apply function to non-offset time array to get rec_dat without latency
        aligned=f(np.arange(len(tx)) / fs)
    except ValueError as err:
        #TODO : there is probably a better fix for this but just return rx data with no shift and give a warning
        warnings.warn(f'Problem during time alignment \'{str(err)}\' returning data with no shift',RuntimeWarning)
        #there was a problem with try our best...
        aligned=rx[:len(tx)]
        
    return aligned

        
        
class PSuD:

    #on load conversion to datetime object fails for some reason
    #TODO : figure out how to fix this, string works for now but this should work too:
    #row[k]=datetime.datetime.strptime(row[k],'%d-%b-%Y_%H-%M-%S')
    data_fields={"Timestamp":str,"Filename":str,"m2e_latency":float,"good_M2E":(lambda s: bool(strtobool(s))),"Over_runs":int,"Under_runs":int}
    no_log=('y','clipi','data_dir','wav_data_dir','csv_data_dir','cutpoints','data_fields','time_expand_samples','num_keywords')
    
    def __init__(self,
                 audioFiles=[],
                 audioPath = '',
                 overPlay=1.0,
                 trials = 100,
                 blockSize=512,
                 bufSize=20,
                 outdir='',
                 ri=None,
                 info=None,
                 fs = 48e3,
                 ptt_wait=0.68,
                 ptt_gap=3.1,
                 rng=np.random.default_rng(),
                 audioInterface=None,
                 mrt= ABC_MRT16(),
                 time_expand = [100e-3 - 0.11e-3, 0.11e-3],
                 m2e_min_corr = 0.76,
                 get_post_notes = None,
                 intell_est='trial'):
        #set default values
        self.audioFiles=audioFiles
        self.audioPath=audioPath
        self.overPlay=overPlay
        self.trials=trials
        self.blockSize=blockSize
        self.bufSize=bufSize
        self.outdir=outdir
        self.ri=ri
        self.info=info
        self.fs=fs
        self.ptt_wait=ptt_wait
        self.ptt_gap=ptt_gap
        self.rng=rng
        self.audioInterface=audioInterface
        self.mrt = mrt
        self.time_expand=time_expand
        self.m2e_min_corr=m2e_min_corr
        self.get_post_notes=get_post_notes
        self.intell_est=intell_est
        
    #load audio files for use in test
    def load_audio(self):
    
        #check that audio files is not empty
        if not self.audioFiles:
            #TODO : is this the right error to use here??
            raise ValueError('Expected self.audioFiles to not be empty')

        #list for input speech
        self.y=[]
        #list for cutpoints
        self.cutpoints=[]
        #list for word spacing
        self.keyword_spacings=[]
        
        for f in self.audioFiles:
            #make full path from relative paths
            f_full=os.path.join(self.audioPath,f)
            # load audio
            fs_file, audio_dat = scipy.io.wavfile.read(f_full)
            #check fs
            if(fs_file != self.fs):
                raise RuntimeError(f'Expected fs to be {self.fs} but got {fs_file} for {f}')
            # Convert to float sound array and add to list
            self.y.append( mcvqoe.audio_float(audio_dat))
            #strip extension from file
            fne,_=os.path.splitext(f_full)
            #add .csv extension
            fcsv=fne+'.csv'
            #load cutpoints
            cp=mcvqoe.load_cp(fcsv)
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
            self.keyword_spacings.append(min(lens)/self.fs)
            
    def set_time_expand(self,t_ex):
        self.time_expand_samples=np.array(t_ex)
        
        if(len(self.time_expand_samples)==1):
            #make symmetric interval
            self.time_expand_samples=np.array([self.time_expand_samples,]*2)

        #convert to samples
        self.time_expand_samples=np.ceil(self.time_expand_samples*self.fs).astype(int)
        
    def audio_clip_check(self):
        #number of keyword columns to have in the .csv file
        self.num_keywords=0
        #check cutpoints and count keywaords
        for cp in self.cutpoints:
            #count the number of actual keywords
            n=sum(not np.isnan(w['Clip']) for w in cp)
            #set num_keywords to max values
            self.num_keywords=max(n,self.num_keywords)
            
    def csv_header_fmt(self):
        hdr=','.join(self.data_fields.keys())
        fmt='{'+'},{'.join(self.data_fields.keys())+'}'
        for word in range(self.num_keywords):
            hdr+=f',W{word}_Int'
            fmt+=f',{{intel[{word}]}}'
        #add newlines at the end
        hdr+='\n'
        fmt+='\n'
        
        return (hdr,fmt)
    
    def run(self):
        #---------------------------[Set time expand]---------------------------
        self.set_time_expand(self.time_expand)
        #---------------------[Load Audio Files if Needed]---------------------
        if(not hasattr(self,'y')):
            self.load_audio()
        
        #generate clip index
        self.clipi=self.rng.permutation(self.trials)%len(self.y)
        
        self.audio_clip_check()
        
        #-------------------[Find and Setup Audio interface]-------------------
        dev=self.audioInterface.find_device()
        
        #set device
        self.audioInterface.device=dev
        
        #set parameteres
        self.audioInterface.buffersize=self.bufSize
        self.audioInterface.blocksize=self.blockSize
        self.audioInterface.overPlay=self.overPlay

        #-------------------------[Get Test Start Time]-------------------------
        self.info['Tstart']=datetime.datetime.now()
        dtn=self.info['Tstart'].strftime('%d-%b-%Y_%H-%M-%S')
        
        #--------------------------[Fill log entries]--------------------------
        self.info.update(mcvqoe.write_log.fill_log(self))
        
        self.info['test']='PSuD'
        
        #-----------------------[Setup Files and folders]-----------------------
        
        #generate data dir names
        data_dir=os.path.join(self.outdir,'data')
        wav_data_dir=os.path.join(data_dir,'wav')
        csv_data_dir=os.path.join(data_dir,'csv')
        
        
        #create data directories 
        os.makedirs(csv_data_dir, exist_ok=True)
        os.makedirs(wav_data_dir, exist_ok=True)
        
        
        #generate base file name to use for all files
        base_filename='capture_%s_%s'%(self.info['Test Type'],dtn);
        
        #generate test dir names
        wavdir=os.path.join(wav_data_dir,base_filename) 
        
        #create test dir
        os.makedirs(wavdir, exist_ok=True)
        
        #get name of audio clip without path or extension
        clip_names=[ os.path.basename(os.path.splitext(a)[0]) for a in self.audioFiles]

        #get name of csv files with path and extension
        self.data_filename=os.path.join(csv_data_dir,f'{base_filename}.csv')

        #get name of temp csv files with path and extension
        temp_data_filename = os.path.join(csv_data_dir,f'{base_filename}_TEMP.csv')

        #write out Tx clips and cutpoints to files
        for dat,name,cp in zip(self.y,clip_names,self.cutpoints):
            out_name=os.path.join(wavdir,f'Tx_{name}')
            wav.write(out_name+'.wav', int(self.fs), dat)
            mcvqoe.write_cp(out_name+'.csv',cp)
            
        #---------------------------[write log entry]---------------------------
        
        mcvqoe.write_log.pre(info=self.info, outdir=self.outdir)
        
        #---------------[Try block so we write notes at the end]---------------
        
        try:
            #---------------------------[Turn on RI LED]---------------------------
            
            self.ri.led(1,True)
            
            #-------------------------[Generate csv header]-------------------------
            
            header,dat_format=self.csv_header_fmt()
            
            #-----------------------[write initial csv file]-----------------------
            with open(temp_data_filename,'wt') as f:
                f.write(header)
            #--------------------------[Measurement Loop]--------------------------
            for trial in range(self.trials):
                #-----------------------[Print Check]-------------------------
                if(trial % 10 == 0):
                    print('-----Trial {}'.format(trial))
                #-----------------------[Get Trial Timestamp]-----------------------
                ts=datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
                #--------------------[Key Radio and play audio]--------------------
                
                #push PTT
                self.ri.ptt(True)
                
                #pause for access
                time.sleep(self.ptt_wait)
                
                clip_index=self.clipi[trial]
                
                #generate filename
                clip_name=os.path.join(wavdir,f'Rx{trial+1}_{clip_names[clip_index]}.wav')
                
                #play/record audio
                self.audioInterface.play_record(self.y[clip_index],clip_name)
                
                #un-push PTT
                self.ri.ptt(False)
                #-----------------------[Pause Between runs]-----------------------
                
                time.sleep(self.ptt_gap)
                
                #-------------------------[Process Audio]-------------------------
                
                #check if we should process audio
                if(self.intell_est=='trial'):
                    trial_dat=self.process_audio(clip_index,clip_name)
                else:
                    #skip processing and give dummy values
                    success=np.empty(self.num_keywords)
                    success.fill(np.nan)
                    #return dummy values to fill in the .csv for now
                    trial_dat={'m2e_latency':None,'intel':success,'good_M2E':False}

                #---------------------------[Write File]---------------------------
                
                trial_dat['Filename']   = clip_names[self.clipi[trial]]
                trial_dat['Timestamp']  = ts
                trial_dat['Over_runs']  = 0
                trial_dat['Under_runs'] = 0
                
                with open(temp_data_filename,'at') as f:
                    f.write(dat_format.format(**trial_dat))
                    
            #-------------------------------[Cleanup]-------------------------------
            
            if(self.intell_est=='post'):
                #process audio from temp file into real file
                print('processing test data')
                
                #load temp file data
                test_dat=self.load_test_data(temp_data_filename,load_audio=False)
                
                #process data and write to final filename
                self.post_process(test_dat,self.data_filename,wavdir)
                
                #remove temp file
                os.remove(temp_data_filename)
            else:
                #move temp file to real file
                shutil.move(temp_data_filename,self.data_filename)
            
            #---------------------------[Turn off RI LED]---------------------------
            
            self.ri.led(1,False)
        
        finally:
            if(self.get_post_notes):
                #get notes
                info=self.get_post_notes()
            else:
                info={}
            #finish log entry
            mcvqoe.post(outdir=self.outdir,info=info)
            
        print(f'Test complete data saved in \'{self.data_filename}\'')
            
        return(base_filename)
        
    def process_audio(self,clip_index,fname):
        
        #---------------------[Load in recorded audio]---------------------
        fs,rec_dat = scipy.io.wavfile.read(fname)
        if(self.fs != fs):
            raise RuntimeError('Recorded sample rate does not match!')
            
        #------------------------[calculate M2E]------------------------
        dly_res = mcvqoe.ITS_delay_est(self.y[clip_index], rec_dat, "f", fsamp=self.fs,min_corr=self.m2e_min_corr)
        
        if(not np.any(dly_res)):
            #M2E estimation did not go super well, try again but restrict M2E bounds to keyword spacing
            dly_res = mcvqoe.ITS_delay_est(self.y[clip_index], rec_dat, "f", fsamp=self.fs,dlyBounds=(0,self.keyword_spacings[clip_index]))
            
            good_m2e=False
        else:
            good_m2e=True
             
        estimated_m2e_latency=dly_res[1] / self.fs

        #---------------------------[align audio]---------------------------
        
        rec_dat_no_latency = align_audio(self.y[clip_index],rec_dat,estimated_m2e_latency,self.fs)
            
        #---------------------[Compute intelligibility]---------------------
        
        success=self.compute_intellligibility(rec_dat_no_latency,self.cutpoints[clip_index])

            
        return {'m2e_latency':estimated_m2e_latency,'intel':success,'good_M2E':good_m2e}

    def compute_intellligibility(self,audio,cutpoints):
        #----------------[Cut audio and perform time expand]----------------

        #array of audio data for each word
        word_audio=[]
        #array of word numbers
        word_num=[]
        #maximum index
        max_idx=len(audio)-1
        
        for cpw in cutpoints:
            if(not np.isnan(cpw['Clip'])):
                #calcualte start and end points
                start=np.clip(cpw['Start']-self.time_expand_samples[0],0,max_idx)
                end  =np.clip(cpw['End']  +self.time_expand_samples[1],0,max_idx)
                #add word audio to array
                word_audio.append(audio[start:end])
                #add word num to array
                word_num.append(cpw['Clip'])

        #---------------------[Compute intelligibility]---------------------
        phi_hat,success=self.mrt.process(word_audio,word_num)
        
        #expand success so len is num_keywords
        success_pad=np.empty(self.num_keywords)
        success_pad.fill(np.nan)
        success_pad[:success.shape[0]]=success
        
        return success_pad
        
    def load_test_data(self,fname,load_audio=True):
            
        with open(fname,'rt') as csv_f:
            #create dict reader
            reader=csv.DictReader(csv_f)
            #create empty list
            data=[]
            #create set for audio clips
            clips=set()
            for row in reader:
                #convert values proper datatype
                for k in row:
                    #check for clip name
                    if(k=='Filename'):
                        #save clips
                        clips.add(row[k])
                    try:
                        #check for None field
                        if(row[k]=='None'):
                            #handle None correcly
                            row[k]=None
                        else:
                            #convert using function from data_fields
                            self.data_fields[k](row[k])
                    except KeyError:
                        #not in data_fields, convert to float
                        row[k]=float(row[k]);
                        
                #append row to data
                data.append(row)
        
        #check if we should load audio
        if(load_audio):
            #set audio file names to Tx file names
            self.audioFiles=['Tx_'+name+'.wav' for name in clips]
            
            dat_name,_=os.path.splitext(os.path.basename(fname))
            
            #set audioPath based on filename
            self.audioPath=os.path.join(os.path.dirname(os.path.dirname(fname)),'wav',dat_name)
            
            #load audio data from files
            self.load_audio()
            self.audio_clip_check()
        
        return data
        
    #get the clip index given a partial clip name
    def find_clip_index(self,name):
        #get all matching indicies
        match=[idx for idx,clip in enumerate(self.audioFiles) if name in clip]
        #check that a match was found
        if(not match):
            raise RuntimeError(f'no audio clips found matching \'{name}\' found in {self.audioFiles}')
        #check that only one match was found
        if(len(match)!=1):
            raise RuntimeError(f'multiple audio clips found matching \'{name}\' found in {self.audioFiles}')
        #return matching index
        return match[0]
        
    def post_process(self,test_dat,fname,audio_path):

        #get .csv header and data format
        header,dat_format=self.csv_header_fmt()
        
        with open(fname,'wt') as f_out:

            f_out.write(header)

            for n,trial in enumerate(test_dat):
                
                #find clip index
                clip_index=self.find_clip_index(trial['Filename'])
                #create clip file name
                clip_name='Rx'+str(n+1)+'_'+trial['Filename']+'.wav'
                
                new_dat=self.process_audio(clip_index,os.path.join(audio_path,clip_name))
                
                #overwrite new data with old and merge
                merged_dat={**trial, **new_dat}

                #write line with new data
                f_out.write(dat_format.format(**merged_dat))


# %%---------------------------------[main]-----------------------------------
if __name__ == "__main__":

    #---------------------------[Create Test object]---------------------------

    #create object here to use default values for arguments
    test_obj=PSuD()
    #set end notes function
    test_obj.get_post_notes=mcvqoe.post_test

    #-----------------------[Setup ArgumentParser object]-----------------------

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
                        '-a', '--audioFiles', default=[],action="extend", nargs="+", type=str,metavar='FILENAME',
                        help='Path to audio files to use for test. Cutpoint files must also be present')
    parser.add_argument(
                        '-f', '--audioPath', default=test_obj.audioPath, type=str,
                        help='Path to look for audio files in. All audio file paths are relative to this unless they are absolute')
    parser.add_argument('-t', '--trials', type=int, default=test_obj.trials,metavar='T',
                        help='Number of trials to use for test. Defaults to %(default)d')
    parser.add_argument("-r", "--radioport", default="",metavar='PORT',
                        help="Port to use for radio interface. Defaults to the first"+
                        " port where a radio interface is detected")
    parser.add_argument('-b', '--blockSize', type=int, default=test_obj.blockSize,metavar='SZ',
                        help='Block size for transmitting audio (default: %(default)d)')
    parser.add_argument('-q', '--bufferSize', type=int, default=test_obj.bufSize,dest='bufSize',metavar='SZ',
                        help='Number of blocks used for buffering audio (default: %(default)d)')
    parser.add_argument('-p', '--overPlay', type=float, default=test_obj.overPlay,metavar='DUR',
                        help='The number of seconds to play silence after the audio is complete'+
                        '. This allows for all of the audio to be recorded when there is delay'+
                        ' in the system')
    parser.add_argument('-x', '--time-expand', type=float, default=test_obj.time_expand,metavar='DUR',dest='time_expand',nargs='+',
                        help='Time in seconds of audio to add before and after keywords before '+
                        'sending them to ABC_MRT. Can be one value for a symmetric expansion or '+
                        'two values for an asymmetric expansion')
                        
    parser.add_argument('--intell-est', default=test_obj.intell_est,dest='intell_est',action='store_const',const='trial',
                        help='Compute intelligibility estimation for audio at end of each trial')
    parser.add_argument('--post-intell-est',dest='intell_est',action='store_const',const='post',
                        help='Compute intelligibility on audio after test is complete')
    parser.add_argument('--no-intell-est',dest='intell_est',action='store_const',const='none',
                        help='don\'t compute intelligibility for audio')
    parser.add_argument('-o', '--outdir', default='', metavar='DIR',
                        help='Directory that is added to the output path for all files')
    parser.add_argument('-w', '--PTTWait', type=float, default=test_obj.ptt_wait, metavar='T',dest='ptt_wait',
                        help='Time to wait between pushing PTT and playing audio')
    parser.add_argument('-g', '--PTTGap', type=float, default=test_obj.ptt_gap, metavar='GAP',dest='ptt_gap',
                        help='Time to pause between trials')
    parser.add_argument('--m2e-min-corr', type=float, default=test_obj.m2e_min_corr, metavar='C',dest='m2e_min_corr',
                        help='Minimum correlation value for acceptable mouth 2 ear measurement (default: %(default)0.2f)')
                                                
                        
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #check that time expand is not too long
    if(len(args.time_expand)>2):
        raise ValueError('argument --time-expand takes only one or two arguments')
    
    #set object properties that exist
    for k,v in vars(args).items():
        if hasattr(test_obj,k):
            setattr(test_obj,k,v)
            
    #-------------------------[Create audio interface]-------------------------
    
    test_obj.audioInterface=mcvqoe.AudioPlayer()
            
    #------------------------------[Get test info]------------------------------
    test_obj.info=mcvqoe.pretest(args.outdir)
    
    #---------------------------[Open RadioInterface]---------------------------
    
    with mcvqoe.RadioInterface(args.radioport) as test_obj.ri:
        #------------------------------[Run Test]------------------------------
        test_obj.run()
