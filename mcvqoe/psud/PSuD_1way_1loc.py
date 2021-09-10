#!/usr/bin/env python

import argparse
import os

from mcvqoe.psud import PSuD

import mcvqoe.gui
import mcvqoe.hardware

def main():
    #---------------------------[Create Test object]---------------------------

    #create object here to use default values for arguments
    test_obj=PSuD.measure()
    #set end notes function
    test_obj.get_post_notes=mcvqoe.gui.post_test
            
    #-------------------------[Create audio interface]-------------------------
    ap=mcvqoe.hardware.AudioPlayer()
    test_obj.audio_interface=ap

    #-----------------------[Setup ArgumentParser object]-----------------------

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
                        '-a', '--audio-files', default=test_obj.audio_files,
                        action="extend", nargs="+", type=str,metavar='FILENAME',
                        help='Path to audio files to use for test. Cutpoint '
                        + 'files must also be present')
    parser.add_argument(
                        '-f', '--audio-path', default=test_obj.audio_path, type=str,
                        help='Path to look for audio files in. All audio file '
                        + 'paths are relative to this unless they are absolute')
    parser.add_argument('-t', '--trials', type=int, default=test_obj.trials,metavar='T',
                        help='Number of trials to use for test. Defaults to %(default)d')
    parser.add_argument("-r", "--radioport", default="",metavar='PORT',
                        help="Port to use for radio interface. Defaults to the first"+
                        " port where a radio interface is detected")
    parser.add_argument('-b', '--blocksize', type=int, default=ap.blocksize,metavar='SZ',
                        help='Block size for transmitting audio (default: %(default)d)')
    parser.add_argument('-q', '--buffersize', type=int, default=ap.buffersize,metavar='SZ',
                        help='Number of blocks used for buffering audio (default: %(default)d)')
    parser.add_argument('-p', '--overplay', type=float, default=ap.overplay,metavar='DUR',
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
    parser.add_argument('-F','--full-audio-dir',dest='full_audio_dir',
                        action='store_true',default=test_obj.full_audio_dir,
                        help='ignore --audioFiles and use all files in --audioPath')
    parser.add_argument('--no-full-audio-dir',dest='full_audio_dir',action='store_false',
                        help='use --audioFiles to determine which audio clips to read')
    parser.add_argument('--save-tx-audio', dest='save_tx_audio',
                        action='store_true',
                        help='Save transmit audio in wav directory')
    parser.add_argument('--no-save-tx-audio', dest='save_tx_audio',
                        action='store_false',
                        help='Don\'t save transmit audio in wav directory')
    parser.add_argument('--save-audio', dest='save_audio', action='store_true',
                        help='Save audio in the wav directory')
    parser.add_argument('--no-save-audio', dest='save_audio', action='store_false',
                        help='Don\'t save audio in the wav directory, implies'+
                        '--no-save-tx-audio')                                                
    parser.add_argument('--audio-set',
                        default=None,
                        type=int,
                        help='PsuD audio set to use for test. Integer between '
                        + '1-4')
    parser.add_argument('-T', '--p-thresh', type=float, dest="p_thresh", default=test_obj.p_thresh,
                        help="The threshold of A-weight power for words, in dB, below which they"+
                        " are considered to have no audio. (Defaults to %(default).1f dB)")
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #check that time expand is not too long
    if(len(args.time_expand)>2):
        raise ValueError('argument --time-expand takes only one or two arguments')
    
    #set object properties that exist
    for k,v in vars(args).items():
        if hasattr(test_obj,k):
            setattr(test_obj,k,v)
    # Update with audio set if passed
    if args.audio_set is not None:
        if 1 <= args.audio_set and args.audio_set <= len(test_obj._default_audio_sets):
            audio_set = test_obj._default_audio_sets[args.audio_set - 1]
            test_obj.audio_path = os.path.join(test_obj._default_audio_path, audio_set)
            test_obj.full_audio_dir = True
        else:
            raise ValueError('audio_set must be between 1-4')
    
    #---------------------[Set audio interface properties]---------------------
    test_obj.audio_interface.blocksize=args.blocksize
    test_obj.audio_interface.buffersize=args.buffersize
    test_obj.audio_interface.overplay=args.overplay
    
    #set correct audio channels
    test_obj.audio_interface.playback_chans={'tx_voice':0}
    test_obj.audio_interface.rec_chans={'rx_voice':0}
    
    #---------------------------[Open RadioInterface]---------------------------
    
    with mcvqoe.hardware.RadioInterface(args.radioport) as test_obj.ri:
                                                    
        #------------------------------[Get test info]------------------------------
        test_obj.info=mcvqoe.gui.pretest(args.outdir,
                    check_function=lambda : mcvqoe.hardware.single_play(
                                                    test_obj.ri,test_obj.audio_interface,
                                                    ptt_wait=test_obj.ptt_wait))
        #------------------------------[Run Test]------------------------------
        test_obj.run()
        print(f'Test complete, data saved in \'{test_obj.data_filename}\'')


# %%---------------------------------[main]-----------------------------------
if __name__ == "__main__":
    main()
