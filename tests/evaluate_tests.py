# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:39:59 2021

@author: jkp4
"""

import os
import unittest
import pdb #TODO: delete this
import sys 

import numpy as np
import pandas as pd

# TODO: Update this when we have a class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PSuD_eval import evaluate

class EvaluateTest(unittest.TestCase):
    data_path = os.path.join('test_data','raw_data')
    test_path = os.path.join('test_data','reference')
    
    technologies = ['analog_direct',
                         'p25_direct',
                         'p25_trunked_P1',
                         'p25_trunked_P2']
    reference_data = {}
    proc_tests = {}
    # ad_path = os.path.join(test_path, 'analogdirect.csv')
    # analog_direct_results = pd.read_csv(ad_path)
    # p25d_path = os.path.join(test_path, 'P25direct.csv')
    # p25_direct_results = pd.read_csv(p25d_path)
    # p25tp1_path = os.path.join(test_path, 'p25TrunkedP1.csv')
    # p25_tp1_results = pd.read_csv(p25tp1_path)
    # p25tp2_path = os.path.join(test_path, 'p25TrunkedP2.csv')
    # p25_tp2_results = pd.read_csv(p25tp2_path)
    
    intelligibility_thresholds = [0.5, 0.7, 0.9]
    message_lengths = [1,3,5,10]
    fs = int(48e3)
    
    @classmethod
    def setUpClass(cls):
        cls.load_reference_data()
        cls.process_data()
    
    @classmethod
    def load_reference_data(cls):
        for tech in cls.technologies:
            tech_file = tech + ".csv"
            tech_path= os.path.join(cls.test_path,tech_file)
            cls.reference_data[tech] = pd.read_csv(tech_path)
        
    @classmethod
    def process_data(cls):
        
        for tech in cls.technologies:
            test_dir = os.path.join(cls.data_path,tech)
            cut_dir = os.path.join(test_dir,'wav')
            test_names = os.listdir(cut_dir)
            
            cls.proc_tests[tech] = evaluate(test_names,
                          test_path= test_dir,
                          fs = cls.fs,
                          use_reprocess=True)
            
    @classmethod
    def no_weight_test(cls,tech_name, method):
        
        ref_data = cls.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in cls.intelligibility_thresholds:
            for mlen in cls.message_lengths:
                psud,psud_ci = cls.proc_tests[tech_name].eval_psud(thresh,
                                                                    mlen,
                                                                    method = method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh) & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0],'PSuD']
                cls.assertAlmostEqual(cls,
                                      psud,
                                       refdat,
                                       7,
                                       'Mismatch for intelligibility threshold of {} and message length {}'.format(thresh,mlen))
    
    @classmethod
    def weight_test(cls,tech_name,method,mweight):
        ref_data = cls.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in cls.intelligibility_thresholds:
            for mlen in cls.message_lengths:
                psud,psud_ci = cls.proc_tests[tech_name].eval_psud(thresh,
                                                                    mlen,
                                                                    method = method,
                                                                    method_weight = mweight)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh) & (tech_ref['Length'] == mlen) & (tech_ref['Weight'] == mweight)]
                tix = tdat.index
                refdat = tdat.loc[tix[0],'PSuD']
                cls.assertAlmostEqual(cls,
                                      psud,
                                        refdat,
                                        7,
                                        'Mismatch for intelligibility threshold of {} and message length {}'.format(thresh,mlen))
            
    def test_analog_direct_ewc(self):
        tech_name = 'analog_direct'
        method = 'EWC'
        
        self.no_weight_test(tech_name, method)
        
    def test_analog_direct_arf(self):
        tech_name = 'analog_direct'
        method = 'ARF'
        mweight = 0.5
        
        self.weight_test(tech_name, method,mweight)
        
    def test_analog_direct_ami(self):
        tech_name = 'analog_direct'
        method = 'AMI'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_direct_ewc(self):
        tech_name = 'p25_direct'
        method = 'EWC'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_direct_arf(self):
        tech_name = 'p25_direct'
        method = 'ARF'
        mweight = 0.5
        
        self.weight_test(tech_name, method, mweight)
        
    
    def test_p25_direct_ami(self):
        tech_name = 'p25_direct'
        method = 'AMI'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_trunked_P1_ewc(self):
        tech_name = 'p25_trunked_P1'
        method = 'EWC'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_trunked_P1_arf(self):
        tech_name = 'p25_trunked_P1'
        method = 'ARF'
        mweight = 0.5
        
        self.weight_test(tech_name, method,mweight)
    
    def test_p25_trunked_P1_ami(self):
        tech_name = 'p25_trunked_P1'
        method = 'AMI'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_trunked_P2_ewc(self):
        tech_name = 'p25_trunked_P2'
        method = 'EWC'
        
        self.no_weight_test(tech_name, method)
        
    def test_p25_trunked_P2_arf(self):
        tech_name = 'p25_trunked_P2'
        method = 'ARF'
        mweight = 0.5
        
        self.weight_test(tech_name, method,mweight)
    
    def test_p25_trunked_P2_ami(self):
        tech_name = 'p25_trunked_P2'
        method = 'AMI'
        
        self.no_weight_test(tech_name, method)
        
if __name__ == "__main__":
    unittest.main()
