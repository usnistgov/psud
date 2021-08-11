# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:37:43 2021

@author: jkp4
"""
import argparse
import csv
import os
import mcvqoe.psud

data_path = 'test_data'
raw_path = os.path.join(data_path, 'raw_data')
ref_path = os.path.join(data_path, 'reference')

all_technologies = ['analog_direct', 'p25_direct', 'p25_trunked_P1',
                    'p25_trunked_P2']
all_methods = ['EWC', 'AMI']
message_lengths = [1, 3, 5, 10]
threshold = [0.5, 0.7, 0.9]


def generate_reference_data(tech_name, method, overwrite=False):
    """
    Generate reference data for PSuD tests.

    Should not be run unless a PSuD evalaute method has fundamentally changed
    and the reference data needs to be updated.

    Parameters
    ----------
    tech_name : str
        Technology for test.
    method : str
        Method for evaluation.
    overwrite : bool, optional
        Overwrite test/method reference data if it already exists.
        The default is False.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    tech_path = os.path.join(raw_path, tech_name)
    tech_files = os.listdir(os.path.join(tech_path,'wav'))
    
    tproc = mcvqoe.psud.evaluate(tech_files,
                                 test_path=tech_path)
    header = ['Method', 'Threshold', 'Length', 'PSuD', 'PSuD_LB', 'PSuD_UB']

    out_name = f'{tech_name}-{method}.csv'
    out_path = os.path.join(ref_path,out_name)
    os.makedirs(ref_path,exist_ok=True)
    if os.path.exists(out_path):
        if not overwrite:
            raise RuntimeError(
                f'File already exists:\n\t{out_path}.\nSet overwrite to replace'
                )
        else:
            print(f'Overwriting reference data\n\t{out_path}.')
    with open(out_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for mlen in message_lengths:
            for thresh in threshold:
                ps, ci = tproc.eval(threshold=thresh,
                                    msg_len=mlen,
                                    method=method)
                row = [method, thresh, mlen, ps, ci[0], ci[1]]
                csvwriter.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--technology',
                        type=list,
                        nargs='*',
                        action='extend',
                        default=[],
                        help=('Technology(ies) to create reference data for. '
                              'If nothing passed all technologies will be '
                              'used.')
                        )
    parser.add_argument('-m','--method',
                        type=list,
                        nargs='*',
                        action='extend',
                        default=[],
                        help=('Method(s) to create reference data for. If '
                              'nothing passed all methods will be used.')
                        )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite existing reference data')
    args = parser.parse_args()

    if args.technology == []:
        technology = all_technologies
    else:
        technology = []
        for tech in args.technology:
            if tech in all_technologies:
                technology.append(tech)
            else:
                raise ValueError(
                    f'Invalid technology: {tech}. Must be one of:\n{all_technologies}'
                    )
    if args.method == []:
        methods = all_methods
    else:
        methods = []
        for meth in methods:
            if meth in all_methods:
                methods.append(meth)
            else:
                raise ValueError(
                    f'Invalid method: {meth}. Must be one of:\n{all_methods}'
                    )
    for tech in technology:
        for method in methods:
            generate_reference_data(tech_name=tech,
                                    method=method,
                                    overwrite=args.overwrite)