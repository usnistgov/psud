#!/usr/bin/env python

import argparse
import os.path
import scipy.io.wavfile
import csv
import numpy as np
import datetime
import shutil
import time
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav
import warnings

from ITS_delay_est import ITS_delay_est
from abcmrt import ABC_MRT16


if __name__ == "__main__":
    from radioInterface import RadioInterface
    
#read in cutpoints from file
#TODO: move this to common library
def load_cp(fname):
    #field names for cutpoints
    cp_fields=['Clip','Start','End']
    #open cutpoints file
    with open(fname,'rt') as csv_f:
        #create dict reader
        reader=csv.DictReader(csv_f)
        #check for correct fieldnames
        if(reader.fieldnames != cp_fields):
            raise RuntimeError(f'Cutpoint columns do not match {cp_fields}')
        #create empty list
        cp=[]
        #append each line
        for row in reader:
            #convert values to float
            for k in row:
                if(k=='Clip'):
                    #float is needed to represent NaN
                    row[k]=float(row[k])
                    #convert non nan fields to int
                    if(not np.isnan(row[k])):
                        row[k]=int(row[k])
                else:
                    #make indexes zero based
                    row[k]=int(row[k])-1;
                
            #append row to 
            cp.append(row)
        return tuple(cp)
     
#write cutpoints to file
#TODO: move this to common library   
def write_cp(fname,cutpoints):
    #field names for cutpoints
    cp_fields=['Clip','Start','End']
    #open cutpoints file
    with open(fname,'wt',newline='\n', encoding='utf-8') as csv_f:
        #create dict writer
        writer=csv.DictWriter(csv_f, fieldnames=cp_fields)
        #write header row
        writer.writeheader()
        #write each row
        for wcp in cutpoints:
            #convert back to 1 based index
            wcp['Start']+=1
            wcp['End']+=1
            #write each row
            writer.writerow(wcp)
        
#convert audio data to float type with standard scale        
#TODO: move this to comon library
def audio_float(dat):
    if(dat.dtype is np.dtype('uint8')):
        return (dat.astype('float')-128)/128
    if(dat.dtype is np.dtype('int16')):
        return dat.astype('float')/(2**15)
    if(dat.dtype is np.dtype('int32')):
        return dat.astype('float')/(2**31)
    if(dat.dtype is np.dtype('float32')):
        return dat 
    if(dat.dtype is np.dtype('float64')):
        return dat    
    else:
        raise RuntimeError(f'unknown audio type \'{dat.dtype}\'')

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
    
    data_fields=("Timestamp","Filename","m2e_latency","Over_runs","Under_runs")
    
    def __init__(self):
        #set default values
        self.audioFiles=[]
        self.audioPath=''
        self.overPlay=1.0
        self.trials=100
        self.blockSize=512
        self.bufSize=20
        self.outdir=''
        self.ri=None
        self.info=None
        self.fs=48e3
        self.ptt_wait=0.68
        self.ptt_gap=3.1
        self.rng=np.random.default_rng()
        self.play_record_func=None
        self.mrt = ABC_MRT16()
        self.time_expand=[100e-3 - 0.11e-3, 0.11e-3]
        
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
        
        for f in self.audioFiles:
            #make full path from relative paths
            f_full=os.path.join(self.audioPath,f)
            # load audio
            fs_file, audio_dat = scipy.io.wavfile.read(f_full)
            #check fs
            if(fs_file != self.fs):
                raise RuntimeError(f'Expected fs to be {self.fs} but got {fs_file} for {f}')
            # Convert to float sound array and add to list
            self.y.append( audio_float(audio_dat))
            #strip extension from file
            fne,_=os.path.splitext(f_full)
            #add .csv extension
            fcsv=fne+'.csv'
            #load cutpoints
            cp=load_cp(fcsv)
            #add cutpoints to array
            self.cutpoints.append(cp)
            
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
        hdr=','.join(self.data_fields)
        fmt="{timestamp},{name},{m2e},{overrun},{underrun}"
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
        #-------------------------[Get Test Start Time]-------------------------
        self.info['Tstart']=datetime.datetime.now()
        dtn=self.info['Tstart'].strftime('%d-%b-%Y_%H-%M-%S')
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
            write_cp(out_name+'.csv',cp)
            
        #---------------------------[write log entry]---------------------------
        
        #-------------------------[Generate csv header]-------------------------
        
        header,dat_format=self.csv_header_fmt()
        
        #-----------------------[write initial csv file]-----------------------
        with open(temp_data_filename,'wt') as f:
            f.write(header)
        #--------------------------[Measurement Loop]--------------------------
        for trial in range(self.trials):
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
            self.play_record_func(self.y[clip_index],buffersize=self.bufSize, blocksize=self.blockSize,
                out_name=clip_name,overPlay=self.overPlay)
            
            #un-push PTT
            self.ri.ptt(False)
            #-----------------------[Pause Between runs]-----------------------
            
            time.sleep(self.ptt_gap)
            #---------------------[Load in recorded audio]---------------------
            fs,rec_dat = scipy.io.wavfile.read(clip_name)
            if(self.fs != fs):
                raise RuntimeError('Recorded sample rate does not match!')
                
            #------------------------[calculate M2E]------------------------
            estimated_m2e_latency = ITS_delay_est(self.y[clip_index], rec_dat, "f", fsamp=self.fs)[1] / self.fs

            #---------------------------[align audio]---------------------------
            
            rec_dat_no_latency = align_audio(self.y[clip_index],rec_dat,estimated_m2e_latency,self.fs)
            
            #---------------------[Compute intelligibility]---------------------
            
            success=self.compute_intellligibility(rec_dat_no_latency,self.cutpoints[clip_index])

            #---------------------------[Write File]---------------------------
            with open(temp_data_filename,'at') as f:
                f.write(dat_format.format(timestamp=ts,name=clip_names[self.clipi[trial]],m2e=estimated_m2e_latency,intel=success,overrun=0,underrun=0))
                
        #-------------------------------[Cleanup]-------------------------------
        
        #move temp file to real file
        shutil.move(temp_data_filename,self.data_filename)

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
        
    def load_test_data(self,fname):
            
        with open(fname,'rt') as csv_f:
            #create dict reader
            reader=csv.DictReader(csv_f)
            #check for correct fieldnames
            for n,field in enumerate(reader.fieldnames):
                if(n<len(self.data_fields)):
                    #standard column
                    expected_field=self.data_fields[n]
                else:
                    #word column
                    expected_field=f'W{n-len(self.data_fields)}_Int'
                #check for a match
                if(field!=expected_field):
                    raise RuntimeError(f'Got \'{field}\' for column name but expected \'{expected_field}\'')
            #create empty list
            data=[]
            #create set for audio clips
            clips=set()
            for row in reader:
                #convert values to float
                for k in row:
                    #check for timestamp field
                    if(k==self.data_fields[0]):
                        #convert to datetime object
                        #well this fails for some reason, I guess a string is not horrible...
                        #TODO : figure out how to fix this
                        #row[k]=datetime.datetime.strptime(row[k],'%d-%b-%Y_%H-%M-%S')
                        pass
                    #check for clip name field
                    elif(k==self.data_fields[1]):
                        clips.add(row[k])
                    else:
                        #convert to float
                        row[k]=float(row[k]);
                    
                #append row to 
                data.append(row)
        
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


#main function 
if __name__ == "__main__":

    #---------------------------[Create Test object]---------------------------

    #create object here to use default values for arguments
    test_obj=PSuD()

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
    parser.add_argument('-o', '--outdir', default='', metavar='DIR',
                        help='Directory that is added to the output path for all files')
    parser.add_argument('-w', '--PTTWait', default=test_obj.ptt_wait, metavar='T',dest='ptt_wait',
                        help='Time to wait between pushing PTT and playing audio')
    parser.add_argument('-g', '--PTTGap', default=test_obj.ptt_gap, metavar='GAP',dest='ptt_gap',
                        help='Time to pause between trials')
                                                
                        
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #check that time expand is not too long
    if(len(args.time_expand)>2):
        raise ValueError('argument --time-expand takes only one or two arguments')
    
    #set object properties that exist
    for k,v in vars(args).items():
        if hasattr(test_obj,k):
            setattr(test_obj,k,v)
            
    #------------------------------[Get test info]------------------------------
    
    #TESTING : put fake test info here for now
    test_obj.info={}
    test_obj.info['Test Type']='testing'
    
    #---------------------------[Open RadioInterface]---------------------------
    
    with RadioInterface(args.radioport) as test_obj.ri:
        #------------------------------[Run Test]------------------------------
        test_obj.run()
        #TESTING : print out all class properties
        print('Properties for test_obj:')
        for k,v in vars(test_obj).items():
            print(f'\t{k} = {v}')
