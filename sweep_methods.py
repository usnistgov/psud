# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:40:42 2021

@author: jkp4
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime



from itertools import combinations
from matplotlib.legend_handler import HandlerLine2D
from PSuD_eval import evaluate

def sweep(t_proc):
    methods = ["EWC","ARF"]
    thresholds = [0.5, 0.7, 0.9]
    lengths = [1, 3, 5, 10]
    weights = np.arange(0.5,0.9,0.1)
    
    results = pd.DataFrame(columns = ['Method','Weight','Threshold','Length','PSuD','PSuD_LB','PSuD_UB'])
    
    
    ix = 0
    for method in methods:
        
        for t in thresholds:
            print('Method: {}, Threshold: {}'.format(method,t))
            if(method == "EWC"):
                for mlen in lengths:
                    psud_m,psud_ci = t_proc.eval_psud(t,
                                                      mlen,
                                                      method = method)
                    results.loc[ix] = [method, np.nan, t, mlen, psud_m, psud_ci[0], psud_ci[1]]
                    ix+= 1
            else:
                for w in weights:
                    for mlen in lengths:
                        psud_m,psud_ci = t_proc.eval_psud(t,
                                                      mlen,
                                                      method = method,
                                                      method_weight = w)
                        results.loc[ix] = [method, w, t, mlen, psud_m, psud_ci[0], psud_ci[1]]
                        t_proc.clear()
                        ix+= 1

def plot_sweep(results):
    
    methods = np.unique(results['Method'])
    weights = np.unique(results['Weight'][~np.isnan(results['Weight'])])
    lengths = np.unique(results['Length'])
    thresholds = np.unique(results['Threshold'])
    
    # TODO: Delete this later
    lengths = [3]
    for mlen in lengths:
        plots = []
        #TODO: Add axis info
        #TODO: Make points clear on plots
        for method in methods:
            if(method == "EWC"):
                fdf = results.loc[(results['Method'] == method) & (results['Length'] == mlen)]
                label_str = 'EWC'
                p, = plt.plot(fdf['Threshold'],fdf['PSuD'], label= label_str)
                plots.append(p)
            else:
                for w in weights:
                    label_str = '{} - weight {}'.format(method,w)
                    fdf = results.loc[(results['Method'] == method) & (results['Length'] == mlen) & (results['Weight'] == w) ]
                    p, = plt.plot(fdf['Threshold'],fdf['PSuD'],label=label_str)
                    plots.append(p)
        plt.legend(handler_map={plots[0]: HandlerLine2D(numpoints=4)})
    

if(__name__ == "__main__"):
        # Set up argument parser
    parser = argparse.ArgumentParser(
        description = __doc__)
    parser.add_argument('test_names',
                        type = str,
                        nargs = "+",
                        action = "extend",
                        help = "Test names (same as name of folder for wav files)")
    parser.add_argument('-p', '--test-path',
                        default = '',
                        type = str,
                        help = "Path where test data is stored. Must contain wav and csv directories.")
    
    parser.add_argument('-f', '--fs',
                        default = 48e3,
                        type = int,
                        help = "Sampling rate for audio in tests")
    parser.add_argument('-o','--outname',
                        default = None,
                        type = str,
                        help = "Where output data will be saved. If None, will be saved in timestampped something")
        
    args = parser.parse_args()
    
    
    t_proc = evaluate(args.test_names,
                      test_path= args.test_path,
                      fs = args.fs,
                      use_reprocess=True)
    
    results = sweep(t_proc)
                    
                        
                
                    
                
# Test Analog direct:
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'Rcapture_Analog-direct_11-Feb-2021_14-23-10 Rcapture_Analog-direct_11-Feb-2021_11-45-21 Rcapture_Analog-direct_11-Feb-2021_09-51-59 Rcapture_Analog-direct_11-Feb-2021_06-22-06 -p data -m 1 3 5 10 -t 0.5 0.7 0.9 --method ARF')

# Test P25 Direct
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'capture_P25-direct_22-Feb-2021_14-18-29 capture_P25-direct_22-Feb-2021_12-24-20 capture_P25-direct_22-Feb-2021_10-02-11 capture_P25-direct_23-Feb-2021_07-01-38 -p data -m 1 3 5 10 -t 0.5 0.7 0.9 --method ARF')

# Test P25 Trunked Phase 1
# runfile('D:/MCV_671DRDOG/psud/PSuD_eval.py', wdir='D:/MCV_671DRDOG/psud', args = 'capture_P25-Trunked-p1_23-Feb-2021_11-28-27 capture_P25-Trunked-p1_23-Feb-2021_09-03-52 capture_P25-Trunked-p1_24-Feb-2021_07-16-10 capture_P25-Trunked-p1_23-Feb-2021_13-24-39 -p data -t 0.5 0.7 0.9 -m 1 3 5 10 --method ARF')