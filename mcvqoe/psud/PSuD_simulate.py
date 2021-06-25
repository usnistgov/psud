#!/usr/bin/env python

import argparse
import os.path
from mcvqoe.psud import PSuD
from .PSuD_eval import evaluate 
import mcvqoe.simulation
import mcvqoe.gui
import sys
import mcvqoe.hardware


def simulate(channel_tech='clean',
             channel_rate = None,
             channel_m2e=None,
             P_a1 = 1,
             P_a2 = 1,
             P_r = 1,
             pInterval = 1,
             audio_path = '',
             overplay=1.0,
             trials = 100,
             blockSize=512,#TODO: Does sim need this and bufSize
             bufSize=20,
             outdir='',
             info={'Test Type':'simulation','Pre Test Notes':None},
             time_expand = [100e-3 - 0.11e-3, 0.11e-3],
             m2e_min_corr = 0.76,
             intell_est='trial',
             split_audio_dest=None,
             full_audio_dir=False,
             test_id=None):
    """
    

    Parameters
    ----------
    channel_tech : TYPE, optional
        DESCRIPTION. The default is 'clean'.
    channel_rate : TYPE, optional
        DESCRIPTION. The default is None.
    P_a1 : TYPE, optional
        DESCRIPTION. The default is 1.
    P_a2 : TYPE, optional
        DESCRIPTION. The default is 1.
    P_r : TYPE, optional
        DESCRIPTION. The default is 1.
    pInterval : TYPE, optional
        DESCRIPTION. The default is 1.
    audioPath : TYPE, optional
        DESCRIPTION. The default is ''.
    overPlay : TYPE, optional
        DESCRIPTION. The default is 1.0.
    trials : TYPE, optional
        DESCRIPTION. The default is 100.
    blockSize : TYPE, optional
        DESCRIPTION. The default is 512.
    bufSize, optional
        DESCRIPTION. The default is 20.
    outdir : TYPE, optional
        DESCRIPTION. The default is ''.
    info : TYPE, optional
        DESCRIPTION. The default is {'Test Type':'simulation','Pre Test Notes':None}.
    time_expand : TYPE, optional
        DESCRIPTION. The default is [100e-3 - 0.11e-3, 0.11e-3].
    m2e_min_corr : TYPE, optional
        DESCRIPTION. The default is 0.76.
    intell_est : TYPE, optional
        DESCRIPTION. The default is 'trial'.
    split_audio_dest : TYPE, optional
        DESCRIPTION. The default is None.
    full_audio_dir : TYPE, optional
        DESCRIPTION. The default is False.
    testID : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    #------------------------[Sim Settings]-----------------------------
    #create sim object
    sim_obj=mcvqoe.simulation.QoEsim(overplay=overplay)
    #TODO : set sim parameters
    sim_obj.channel_tech=channel_tech
    
    #set channel rate, check for None
    if(channel_rate=='None'):
        sim_obj.channel_rate=None
    else:
        sim_obj.channel_rate=channel_rate
    if(channel_m2e is not None):
        sim_obj.m2e_latency=channel_m2e
    
    #construct string for system name
    system=sim_obj.channel_tech
    if(sim_obj.channel_rate is not None):
        system+=' at '+str(sim_obj.channel_rate)
            
   #-------------------[Create Test object]---------------------------
    
    #create object here to use default values for arguments
    test_obj = PSuD.measure(audio_path = audio_path,
                 trials = trials,
                 outdir=outdir,
                 ri=sim_obj,# handled by sim object
                 info=info,
                 ptt_wait=0,# 0 for sim
                 ptt_gap=0,# 0 for sim
                 audio_interface=sim_obj,# handled by sim object
                 time_expand = time_expand,
                 m2e_min_corr = m2e_min_corr,
                 get_post_notes = lambda : mcvqoe.gui.post_test(error_only=True),#only get test notes on error
                 intell_est=intell_est,
                 split_audio_dest=split_audio_dest,
                 full_audio_dir=full_audio_dir)

    #-------------------------------[Probabilityiser]-------------------------
    prob=mcvqoe.simulation.PBI()
        
    prob.P_a1=P_a1
    prob.P_a2=P_a2
    prob.P_r=P_r
    prob.interval=pInterval
    
    
    #-----------------------------[Log Info]--------------------------

    test_obj.info['codec'] = channel_tech
    test_obj.info['codec-rate'] = channel_rate
    test_obj.info['PBI P_a1']=str(P_a1)
    test_obj.info['PBI P_a2']=str(P_a2)
    test_obj.info['PBI P_r'] =str(P_r)
    test_obj.info['PBI interval']=str(pInterval)
    test_obj.info['test-ID'] = test_id
    
    test_obj.info['test_type'] = "simulation"
    test_obj.info['tx_dev'] = "none"
    test_obj.info['rx_dev'] = "none"
    test_obj.info['system'] = system
    test_obj.info['test_loc'] = "N/A"
    
    sim_obj.pre_impairment=prob.process_audio
    
    #--------------------------------[Run Test]--------------------------------
    test_name = test_obj.run()
    test_path = os.path.join(outdir,"data")
    
    print(f"Test complete. Data stored in {test_path}")
    return(test_path)

def main():
    #---------------------------[Create Test object]---------------------------

    #create sim object
    sim_obj=mcvqoe.simulation.QoEsim()
    #TODO : set sim parameters

    #create object here to use default values for arguments
    test_obj=PSuD.measure()
    #set wait times to zero for simulation
    test_obj.ptt_wait=0
    test_obj.ptt_gap=0
    #only get test notes on error
    test_obj.get_post_notes=lambda : mcvqoe.gui.post_test(error_only=True)
    
    #set audioInterface to sim object
    test_obj.audio_interface=sim_obj
    #set radio interface object to sim object
    test_obj.ri=sim_obj

    #-----------------------[Setup ArgumentParser object]-----------------------

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
                        '-a', '--audio-files', default=[],action="extend", nargs="+", type=str,metavar='FILENAME',
                        help='Path to audio files to use for test. Cutpoint files must also be present')
    parser.add_argument(
                        '-f', '--audio-path', default=test_obj.audio_path, type=str,
                        help='Path to look for audio files in. All audio file paths are relative to this unless they are absolute')
    parser.add_argument('-t', '--trials', type=int, default=test_obj.trials,metavar='T',
                        help='Number of trials to use for test. Defaults to %(default)d')
    parser.add_argument("-r", "--radioport", default="",metavar='PORT',
                        help="Port to use for radio interface. Defaults to the first"+
                        " port where a radio interface is detected")
    parser.add_argument('-p', '--overplay', type=float, default=sim_obj.overplay,metavar='DUR',
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
    parser.add_argument('-c','--channel-tech', default=sim_obj.channel_tech, metavar='TECH',dest='channel_tech',
                        help='Channel technology to simulate (default: %(default)s)')
    parser.add_argument('--channel-rate', default=sim_obj.channel_rate, metavar='RATE',dest='channel_rate',
                        help='Channel technology rate to simulate. Passing \'None\' will use the technology default. (default: %(default)s)')
    parser.add_argument('--channel-m2e', type=float, default=sim_obj.m2e_latency, metavar='L',dest='m2e_latency',
                        help='Channel mouth to ear latency, in seconds, to simulate. (default: %(default)s)')
    parser.add_argument('--msg-eval',
                        type = list,
                        default = [1,5,10],
                        help = "Message lengths to evalue PSuD at upon completion")      
    parser.add_argument('--intell-threshold',
                        type = float,
                        default = 0.5,
                        help = "Intelligibility success threshold") 
    parser.add_argument('-F','--full-audio-dir',dest='full_audio_dir',action='store_true',default=False,
                        help='ignore --audioFiles and use all files in --audioPath')
    parser.add_argument('--no-full-audio-dir',dest='full_audio_dir',action='store_false',
                        help='use --audioFiles to determine which audio clips to read')             
                                                
                        
    #-----------------------------[Parse arguments]-----------------------------

    args = parser.parse_args()
    
    #check that time expand is not too long
    if(len(args.time_expand)>2):
        raise ValueError('argument --time-expand takes only one or two arguments')
    
    #set object properties that exist
    for k,v in vars(args).items():
        if hasattr(test_obj,k):
            setattr(test_obj,k,v)
            

    #-------------------------[Set simulation settings]-------------------------

    sim_obj.channel_tech=args.channel_tech
    
    sim_obj.overplay=args.overplay
    
    #set channel rate, check for None
    if(args.channel_rate=='None'):
        sim_obj.channel_rate=None
    else:
        sim_obj.channel_rate=args.channel_rate
        
    sim_obj.m2e_latency=args.m2e_latency
    
    #set correct channels    
    sim_obj.playback_chans={'tx_voice':0}
    sim_obj.rec_chans={'rx_voice':0}
        
    #------------------------------[Get test info]------------------------------
    
    gui=mcvqoe.gui.TestInfoGui(write_test_info=False)
    
    gui.chk_audio_function=lambda : mcvqoe.hardware.single_play(sim_obj,sim_obj,
                                                    playback=True,
                                                    ptt_wait=test_obj.ptt_wait)

    #construct string for system name
    system=sim_obj.channel_tech
    if(sim_obj.channel_rate is not None):
        system+=' at '+str(sim_obj.channel_rate)

    gui.info_in['test_type'] = "simulation"
    gui.info_in['tx_dev'] = "none"
    gui.info_in['rx_dev'] = "none"
    gui.info_in['system'] = system
    gui.info_in['test_loc'] = "N/A"
    test_obj.info=gui.show()

    #check if the user canceled
    if(test_obj.info is None):
        print(f"\n\tExited by user")
        sys.exit(1)
    
    #---------------------------[add probabilityiesr]---------------------------
    
    if(args.use_probabilityiser):
        #TODO: Move this outside of if so can be used to validate results
        prob=mcvqoe.simulation.PBI()
        
        prob.P_a1=args.P_a1
        prob.P_a2=args.P_a2
        prob.P_r=args.P_r
        prob.interval=args.pInterval
        
        
        test_obj.info['PBI P_a1']=str(args.P_a1)
        test_obj.info['PBI P_a2']=str(args.P_a2)
        test_obj.info['PBI P_r'] =str(args.P_r)
        test_obj.info['PBI interval']=str(args.pInterval)
        
        sim_obj.pre_impairment=prob.process_audio
    
    
    #--------------------------------[Run Test]--------------------------------
    test_name = test_obj.run()
    print(f'Test complete, data saved in \'{test_obj.data_filename}\'')
    test_path = os.path.join(args.outdir,"data")
    #--------------------------------[Evaluate Test]---------------------------
    # TODO: Make this fs determination smarter
    t_proc = evaluate(test_name,
                          test_path = test_path,
                          fs = 48e3)
    print("----Intelligibility Success threshold = {}----".format(args.intell_threshold))
    print("Results shown as Psud(t) = mean, (95% C.I.)")
    
    for msg_len in args.msg_eval:
            psud_m,psud_ci = t_proc.eval_psud(args.intell_threshold,msg_len)
            
            if(args.use_probabilityiser):
                e_psud = prob.expected_psud(msg_len)
                if(psud_ci[0] <= e_psud and e_psud <= psud_ci[1]):
                    match = True
                else:
                    match = False
                results_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f}) | Expected = {:.4f} | Pass: {}"
                results = results_str.format(msg_len,
                                             psud_m,
                                             psud_ci[0],
                                             psud_ci[1],
                                             e_psud,
                                             match)
            else:
                results_str = "PSuD({}) = {:.4f}, ({:.4f},{:.4f})"
                results = results_str.format(msg_len,
                                    psud_m,
                                    psud_ci[0],
                                    psud_ci[1])
            print(results)
    #TESTING : print out all class properties
    #print('Properties for test_obj:')
    #for k,v in vars(test_obj).items():
    #    print(f'\t{k} = {v}')

#-----------------------------[main function]-----------------------------
if __name__ == "__main__":
    main()