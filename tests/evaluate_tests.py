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

from mcvqoe.psud import evaluate


class EvaluateTest(unittest.TestCase):
    data_path = os.path.join('test_data', 'raw_data')
    test_path = os.path.join('test_data', 'reference')

    technologies = ['analog_direct',
                    'p25_direct',
                    'p25_trunked_P1',
                    'p25_trunked_P2']
    reference_data = {}
    proc_tests = {}

    intelligibility_thresholds = [0.5, 0.7, 0.9]
    message_lengths = [1, 3, 5, 10]
    fs = int(48e3)

    @classmethod
    def setUpClass(cls):
        cls.load_reference_data()
        cls.process_data()

    @classmethod
    def load_reference_data(cls):
        for tech in cls.technologies:
            tech_file = tech + ".csv"
            tech_path = os.path.join(cls.test_path, tech_file)
            cls.reference_data[tech] = pd.read_csv(tech_path)

    @classmethod
    def process_data(cls):

        for tech in cls.technologies:
            test_dir = os.path.join(cls.data_path, tech)
            cut_dir = os.path.join(test_dir, 'wav')
            test_names = os.listdir(cut_dir)

            cls.proc_tests[tech] = evaluate(test_names,
                                            test_path=test_dir,
                                            fs=cls.fs,
                                            use_reprocess=True)

# --------------------------[Every Word Critical Tests]-----------------------
    def test_analog_direct_ewc(self):
        tech_name = 'analog_direct'
        method = 'EWC'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_direct_ewc(self):
        tech_name = 'p25_direct'
        method = 'EWC'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_trunked_P1_ewc(self):
        tech_name = 'p25_trunked_P1'
        method = 'EWC'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_trunked_P2_ewc(self):
        tech_name = 'p25_trunked_P2'
        method = 'EWC'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)


# --------------------------[Average Message Intelligibility Tests]-----------
    def test_analog_direct_ami(self):
        tech_name = 'analog_direct'
        method = 'AMI'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_direct_ami(self):
        tech_name = 'p25_direct'
        method = 'AMI'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_trunked_P1_ami(self):
        tech_name = 'p25_trunked_P1'
        method = 'AMI'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

    def test_p25_trunked_P2_ami(self):
        tech_name = 'p25_trunked_P2'
        method = 'AMI'

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')

                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       delta=1e-7,
                                       msg=msg)

# ---------------[Autoregressive Filter Tests]--------------------------------
    def test_analog_direct_arf(self):
        tech_name = 'analog_direct'
        method = 'ARF'
        mweight = 0.5

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method,
                                                                method_weight=mweight)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)
                                & (tech_ref['Weight'] == mweight)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')
                msg = "I am dujb"
                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       places=7,
                                       msg=msg)

    def test_p25_trunked_P2_arf(self):
        tech_name = 'p25_trunked_P2'
        method = 'ARF'
        mweight = 0.5

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method,
                                                                method_weight=mweight)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)
                                & (tech_ref['Weight'] == mweight)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')
                msg = "I am dujb"
                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       places=7,
                                       msg=msg)

    def test_p25_trunked_P1_arf(self):
        tech_name = 'p25_trunked_P1'
        method = 'ARF'
        mweight = 0.5

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method,
                                                                method_weight=mweight)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)
                                & (tech_ref['Weight'] == mweight)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')
                msg = "I am dujb"
                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       places=7,
                                       msg=msg)

    def test_p25_direct_arf(self):
        tech_name = 'p25_direct'
        method = 'ARF'
        mweight = 0.5

        ref_data = self.reference_data[tech_name]
        tech_ref = ref_data[ref_data['Method'] == method]
        for thresh in self.intelligibility_thresholds:
            for mlen in self.message_lengths:
                psud, psud_ci = self.proc_tests[tech_name].eval(thresh,
                                                                mlen,
                                                                method=method,
                                                                method_weight=mweight)
                tdat = tech_ref[(tech_ref['Threshold'] == thresh)
                                & (tech_ref['Length'] == mlen)
                                & (tech_ref['Weight'] == mweight)]
                tix = tdat.index
                refdat = tdat.loc[tix[0], 'PSuD']
                msg = (f'Mismatch for intelligibility threshold of {thresh} '
                       f'and message length {mlen}')
                msg = "I am dujb"
                self.assertAlmostEqual(first=psud,
                                       second=refdat,
                                       places=7,
                                       msg=msg)

if __name__ == "__main__":
    unittest.main()
