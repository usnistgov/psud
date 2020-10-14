#!/usr/bin/env python

import argparse
import os.path
import scipy.io.wavfile
import csv
import numpy as np
import datetime

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
                else:
                    #make indexes zero based
                    row[k]=int(row[k])-1;
                
            #append row to 
            cp.append(row)
        return tuple(cp)
        
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

        
        
class PSuD:
    
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
        self.rng=np.random.default_rng()
        
    #load audio files for use in test
    def load(self):
    
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
        
    def run(self):
        #---------------------[Load Audio Files if Needed]---------------------
        if(not hasattr(self,'y')):
            self.load()
        
        #generate clip index
        self.clipi=self.rng.permutation(100)%len(self.y)
        
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
        
        #get name of audio clip without path or extension
        clip_names=[ os.path.basename(os.path.splitext(a)[0]) for a in self.audioFiles]

        #get name of csv files with path and extension
        self.data_filename=os.path.join(csv_data_dir,f'{base_filename}.csv')

        #get name of temp csv files with path and extension
        temp_data_filename = os.path.join(csv_data_dir,f'{base_filename}_TEMP.csv')

        
        #---------------------------[write log entry]---------------------------
        
        #--------------------------[Measurement Loop]--------------------------
        for trial in range(self.trials):
            #-----------------------[Get Trial Timestamp]-----------------------
            timestamp=datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
            #--------------------[Key Radio and play audio]--------------------
            #-----------------------[Pause Between runs]-----------------------
            #------------------------[calculate M2E]------------------------
            #---------------------[Compute intelligibility]---------------------
            #---------------------------[Write Files]---------------------------
            pass



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
    parser.add_argument('-o', '--outdir', default='', metavar='DIR',
                        help='Directory that is added to the output path for all files')
                            
                        
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
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
