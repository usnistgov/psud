#!/usr/bin/env python


import PSuD_1way_1loc
import argparse
import os.path
import scipy.io.wavfile
import csv
import sys

from ITS_delay_est import ITS_delay_est

def reprocess(datafile,outfile=None,test_obj=None):
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
    
    with (os.fdopen(os.dup(sys.stdout.fileno()), 'w') if not outfile else open(outfile, 'w')) as f_out:
        
        f_out.write(header)
        
        for n,trial in enumerate(test_dat):
            
            #find clip index
            clip_index=test_obj.find_clip_index(trial['Filename'])
            #create clip file name
            clip_name='Rx'+str(n+1)+'_'+trial['Filename']+'.wav'
            #load audio file
            fs_file, rec_dat = scipy.io.wavfile.read(os.path.join(test_obj.audioPath,clip_name))
            if(test_obj.fs != fs_file):
                raise RuntimeError('Recorded sample rate does not match!')
                
            #------------------------[calculate M2E]------------------------
            estimated_m2e_latency = ITS_delay_est(test_obj.y[clip_index], rec_dat, "f", fsamp=test_obj.fs)[1] / test_obj.fs

            #---------------------------[align audio]---------------------------
            
            rec_dat_no_latency = PSuD_1way_1loc.align_audio(test_obj.y[clip_index],rec_dat,estimated_m2e_latency,test_obj.fs)
            
            #---------------------[Compute intelligibility]---------------------
            
            success=test_obj.compute_intellligibility(rec_dat_no_latency,test_obj.cutpoints[clip_index])

            #---------------------------[Write File]---------------------------

            f_out.write(dat_format.format(timestamp=trial['Timestamp'],name=trial['Filename'],m2e=estimated_m2e_latency,intel=success,overrun=trial['Over_runs'],underrun=trial['Under_runs']))



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
                                                
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #set time expand
    test_obj.set_time_expand(args.time_expand)
    
    reprocess(args.datafile,args.outfile,test_obj)
    
