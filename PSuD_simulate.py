#!/usr/bin/env python

import argparse
import os.path
import scipy.io.wavfile as wav
import numpy as np
import shutil
from PSuD_1way_1loc import PSuD as PSuD
import mcvqoe.simulation

#main function 
if __name__ == "__main__":

    #---------------------------[Create Test object]---------------------------

    #create sim object
    sim_obj=mcvqoe.simulation.QoEsim()
    #TODO : set sim parameters

    #create object here to use default values for arguments
    test_obj=PSuD()
    #set wait times to zero for simulation
    test_obj.ptt_wait=0
    test_obj.ptt_gap=0
    
    #set play record function to simulated play_record
    test_obj.play_record_func=sim_obj.play_record
    #set radio interface object to sim object
    test_obj.ri=sim_obj

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
    parser.add_argument('-P','--use-probabilityiser', default=False,dest='use_probabilityiser',action='store_true',
                        help='Use probabilityiesr to make channel "flaky"')
    parser.add_argument('--no-use-probabilityiser',dest='use_probabilityiser',action='store_false',
                        help='don\'t use probabilityiesr')
    parser.add_argument('--P-a1',dest='P_a1',type=float,default=1,
                        help='P_a1 for probabilityiesr')
    parser.add_argument('--P-a2',dest='P_a2',type=float,default=1,
                        help='P_a2 for probabilityiesr')
    parser.add_argument('--P-r',dest='P_r',type=float,default=1,
                        help='P_r for probabilityiesr')
    parser.add_argument('--P-interval',dest='pInterval',type=float,default=1,
                        help='Time interval for probabilityiesr in seconds')
    parser.add_argument('--m2e-min-corr', default=test_obj.m2e_min_corr, metavar='C',dest='m2e_min_corr',
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
            
    #---------------------------[add probabilityiesr]---------------------------
    
    if(args.use_probabilityiser):
        prob=mcvqoe.simulation.PBI()
        
        prob.P_a1=args.P_a1
        prob.P_a2=args.P_a2
        prob.P_r=args.P_r
        prob.interval=args.pInterval
        
        sim_obj.pre_impairment=prob.process_audio
        
    #------------------------------[Get test info]------------------------------
    
    #TESTING : put fake test info here for now
    test_obj.info={}
    test_obj.info['Test Type']='testing'
    
    #--------------------------------[Run Test]--------------------------------
    test_obj.run()
    #TESTING : print out all class properties
    #print('Properties for test_obj:')
    #for k,v in vars(test_obj).items():
    #    print(f'\t{k} = {v}')
