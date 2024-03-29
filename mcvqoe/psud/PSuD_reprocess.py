#!/usr/bin/env python


import argparse
import csv
import mcvqoe
import os.path
import scipy.io.wavfile
import sys
import tempfile

from .PSuD import measure
from .PSuD_eval import evaluate
    
def main():
    
    #---------------------------[Create Test object]---------------------------

    # create object here to use default values for arguments
    test_obj=measure()

    #-----------------------[Setup ArgumentParser object]-----------------------
    
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('datafile', default=None, type=str,
                        help='CSV file from test to reprocess')
    parser.add_argument('outfile', default=None, type=str, nargs='?',
                        help='file to write reprocessed CSV data to. Can be the same name as datafile to overwrite results. if omitted output will be written to stdout')
    parser.add_argument('-x', '--time-expand', type=float, default=test_obj.time_expand, metavar='DUR', dest='time_expand', nargs='+',
                        help='Time in seconds of audio to add before and after keywords before '+
                        'sending them to ABC_MRT. Can be one value for a symetric expansion or '+
                        'two values for an asymmetric expansion')
    parser.add_argument('--audio-path', type=str, default=None, metavar='P', dest='audio_path',
                        help='Path to audio files for test. Will be found automatically if not given')
    parser.add_argument('-s', '--split-audio-folder', default=test_obj.split_audio_dest, type=str, dest='split_audio_dest',
                        help='Folder to store single word clips to')
    parser.add_argument('--msg-eval',
                        type = list,
                        default = [1,5,10],
                        help = "Message lengths to evalue PSuD at upon completion")      
    parser.add_argument('--intell-threshold',
                        type = float,
                        default = 0.5,
                        help = "Intelligibility success threshold")
    parser.add_argument('-T', '--p-thresh', type=float, dest="p_thresh", default=test_obj.p_thresh,
                        help="The threshold of A-weight power for words, in dB, below which they"+
                        " are considered to have no audio. (Defaults to %(default).1f dB)")
    
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    test_obj.split_audio_dest = args.split_audio_dest

    # set time expand
    test_obj.time_expand = args.time_expand

    # set power threshold
    test_obj.p_thresh = args.p_thresh
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        if (args.outfile == '--'):
            # print results, don't save file
            out_name = os.path.join(tmp_dir, 'tmp.csv')
            print_outf = True
        elif (args.outfile):
            out_name = args.outfile
            print_outf = False
        else:
            # split data file path into parts
            d, n = os.path.split(args.datafile)
            # construct new name for file
            out_name = os.path.join(d, 'R'+n)
            print_outf = False

        print(f'Loading test data from \'{args.datafile}\'', file=sys.stderr)
        
        # read in test data
        test_dat = test_obj.load_test_data(args.datafile, audio_path=args.audio_path)

        print(f'Reprocessing test data to \'{out_name}\'', file=sys.stderr)
            
        test_obj.post_process(test_dat, out_name, test_obj.audio_path)
            
        if (print_outf):
            with open(out_name, 'rt') as out_file:
                dat = out_file.read()
            print(dat)
            
        print(f'Reprocessing complete for \'{out_name}\'', file=sys.stderr)
        
        #--------------------------------[Evaluate Test]---------------------------
        
        # TODO: Make this fs determination smarter
        t_proc = evaluate(out_name, wav_dirs=test_obj.audio_path, fs=48e3)
        print("----Intelligibility Success threshold = {}----".format(args.intell_threshold))
        print("Results shown as Psud(t) = mean, (95% C.I.)")
        
        for msg_len in args.msg_eval:
                psud_m, psud_ci = t_proc.eval(args.intell_threshold, msg_len)
            
                results_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
                results = results_str.format(msg_len,
                                    psud_m,
                                    psud_ci[0],
                                    psud_ci[1])
                print(results)

# main function 
if __name__ == "__main__":
    main()