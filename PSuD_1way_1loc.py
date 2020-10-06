#!/usr/bin/env python

import argparse


if __name__ == "__main__":
    from radioInterface import RadioInterface
    


class PSuD:
    
    def __init__(self):
        #set default values
        self.audioFiles=[]
        self.audioPath=''
        self.overPlay=1.0
        self.trials=100
        self.blockSize=512
        self.bufSize=20
        self.ri=None
        self.info=None
        
    def run(self):
        #--------------------------[Load Audio Files]--------------------------
        #-------------------[Find and Setup Audio interface]-------------------
        #-----------------------[Setup Files and folders]-----------------------
        #---------------------------[write log entry]---------------------------
        
        #--------------------------[Measurement Loop]--------------------------
        for trial in range(self.trials):
            #-----------------------[Get Trial Timestamp]-----------------------
            #--------------------[Key Radio and play audio]--------------------
            #-----------------------[Pause Between runs]-----------------------
            #------------------------[calculate M2E]------------------------
            #---------------------[Compute intelligibility]---------------------
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
    #---------------------------[Open RadioInterface]---------------------------
    
    with RadioInterface(args.radioport) as test_obj.ri:
        #------------------------------[Run Test]------------------------------
        test_obj.run()
        #TESTING : print out all class properties
        print('Properties for test_obj:')
        for k,v in vars(test_obj).items():
            print(f'\t{k} = {v}')
