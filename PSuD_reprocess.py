#!/usr/bin/env python


import PSuD_1way_1loc
import argparse
import os.path
import scipy.io.wavfile
import csv
import sys
from PSuD_process import PSuD_process 
import tempfile

import mcvqoe

def reprocess(datafile,f_out,test_obj=None):
    #---------------------[Create Test object if not given]---------------------
    
    if(not test_obj):
        test_obj=PSuD_1way_1loc.PSuD()
        #setup time expand values
        test_obj.set_time_expand(test_obj.time_expand)
     
    #TODO : do we really need this?
    #set wait times to zero for reprocess
    test_obj.ptt_wait=0
    test_obj.ptt_gap=0
    
    #set dummy functions for hardware we don't have
    test_obj.play_record_func=None
    test_obj.ri=None

    #---------------------------[Load Data from file]---------------------------

    test_dat=test_obj.load_test_data(datafile)
    
    #-------------------------[Generate csv header]-------------------------
    
    header,dat_format=test_obj.csv_header_fmt()
    
    f_out.write(header)
    
    for n,trial in enumerate(test_dat):
        
        #find clip index
        clip_index=test_obj.find_clip_index(trial['Filename'])
        #create clip file name
        clip_name='Rx'+str(n+1)+'_'+trial['Filename']+'.wav'
        
        new_dat=test_obj.process_audio(clip_index,os.path.join(test_obj.audioPath,clip_name))
        
        #overwrite new data with old and merge
        merged_dat={**trial, **new_dat}
        

        f_out.write(dat_format.format(**merged_dat))
            
    return test_obj.audioPath



#main function 
if __name__ == "__main__":

    #---------------------------[Create Test object]---------------------------

    #create object here to use default values for arguments
    test_obj=PSuD_1way_1loc.PSuD()

    #-----------------------[Setup ArgumentParser object]-----------------------
    
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('datafile', default=None,type=str,
                        help='CSV file from test to reprocess')
    parser.add_argument('outfile', default=None, type=str, nargs='?',
                        help='file to write reprocessed CSV data to. Can be the same name as datafile to overwrite results. if omitted output will be written to stdout')
    parser.add_argument('-o', '--outdir', default='', metavar='DIR',
                        help='Directory that is added to the output path for all files')
    parser.add_argument('-x', '--time-expand', type=float, default=test_obj.time_expand,metavar='DUR',dest='time_expand',nargs='+',
                        help='Time in seconds of audio to add before and after keywords before '+
                        'sending them to ABC_MRT. Can be one value for a symetric expansion or '+
                        'two values for an asymmetric expansion')
    parser.add_argument('--msg-eval',
                        type = list,
                        default = [1,5,10],
                        help = "Message lengths to evalue PSuD at upon completion")      
    parser.add_argument('--intell-threshold',
                        type = float,
                        default = 0.5,
                        help = "Intelligibility success threshold")                                                              
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #set time expand
    test_obj.set_time_expand(args.time_expand)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        if(args.outfile):
            out_name=args.outfile
            print_outf=False
        else:
            out_name=os.path.join(tmp_dir,'tmp.csv')
            print_outf=True
            
        with open(out_name,'wt') as out_file:
            wav_dir=reprocess(args.datafile,out_file,test_obj)
            
        if(print_outf):
            with open(out_name,'rt') as out_file:
                dat=out_file.read()
            print(dat)
            
        
        #--------------------------------[Evaluate Test]---------------------------
        # TODO: Make this fs determination smarter
        t_proc = PSuD_process(out_name,wav_dirs=wav_dir,fs = 48e3)
        print("----Intelligibility Success threshold = {}----".format(args.intell_threshold))
        print("Results shown as Psud(t) = mean, (95% C.I.)")
        
        for msg_len in args.msg_eval:
                psud_m,psud_ci = t_proc.eval_psud(args.intell_threshold,msg_len)
            
                results_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
                results = results_str.format(msg_len,
                                    psud_m,
                                    psud_ci[0],
                                    psud_ci[1])
                print(results)
