import os
import sys
import re
import gzip

import itertools
import numpy as np
from pathlib import Path
import logging
import pandas as pd

import json
import urllib.request

from dev.preprocess.supp_data import *


def unzip_res_range(res_range):
    res_ranges = res_range.strip()[1:-1].split(',')
    index_list = []
    for r in res_ranges:
        if re.match('.+-.+', r):
            a, b = r.split('-')
            index_list += [str(n) for n in range(int(a), int(b)+1)]
        else:
            index_list.append(r)

    if index_list == ['']:
        return []
    else:
        return index_list


def fill_nan_mean(array, axis=0):
    if axis not in [0, 1]:
        raise ValueError('Invalid axis: %s' % axis)
    mean_array = np.nanmean(array, axis=axis)
    inds = np.where(np.isnan(array))
    array[inds] = np.take(mean_array, inds[1-axis])
    if np.any(np.isnan(array)):
        full_array_mean = np.nanmean(array)
        inds = np.unique(np.where(np.isnan(array))[1-axis])
        if axis == 0:
            array[:,inds] = full_array_mean
        else:
            array[inds] = full_array_mean
    return array


def parse_fasta(fasta_file):
    """
    (from Charles)
    Load a fasta file of sequences into a dictionary. Supports both *.fasta and *.fasta.gz.

    Args:
      fasta_file: str, path to the fasta file.

    Returns:
      A dictionary of sequences.
    """
    result_dict = {}
    if fasta_file.endswith('.gz'):
        zipped = True
        f = gzip.open(fasta_file, 'rb')
    else:
        zipped = False
        f = open(fasta_file, 'r')
    seq = ''
    for line in f:
        if zipped:
            is_header = line.startswith(b'>')
        else:
            is_header = line.startswith('>')
        if is_header:
            if seq:
                result_dict[identifier] = seq
            if zipped:
                identifier = line.decode('utf-8').split('|')[1]  # YL: SwissProt ID as key
            else:
                identifier = line.split('|')[1]
            seq = ''
        else:
            if zipped:
                seq += line.decode('utf-8').strip()
            else:
                seq += line.strip()

    result_dict[identifier] = seq
    f.close()
    return result_dict


def parse_fasta_info(fasta_file):
    """
    (Modified from Charles)
    Load protein sequence and definition from FASTA file. Supports both *.fasta and *.fasta.gz.

    Args:
      fasta_file: str, path to the fasta file.

    Returns:
      1) dictionary of sequences, 2) dictionary of protein name (definition)
    """
    seq_dict = {}
    desc_dict = {}
    if fasta_file.endswith('.gz'):
        zipped = True
        f = gzip.open(fasta_file, 'rb')
    else:
        zipped = False
        f = open(fasta_file, 'r')
    seq = ''
    for line in f:
        if zipped:
            is_header = line.startswith(b'>')
        else:
            is_header = line.startswith('>')
        if is_header:
            if seq:
                seq_dict[identifier] = seq
            if zipped:
                info = line.decode('utf-8').split('|')
                # identifier = line.decode('utf-8').split('|')[1]  # YL: SwissProt ID as key
                
            else:
                info = line.split('|')
            seq = ''
            identifier = info[1]
            bound_idx = info[2].find('OS=')  # extract information before species name
            desc = info[2][:bound_idx].strip()
            desc_dict[identifier] = ' '.join(desc.split(' ')[1:])
        else:
            if zipped:
                seq += line.decode('utf-8').strip()
            else:
                seq += line.strip()

    seq_dict[identifier] = seq
    
    f.close()
    return seq_dict, desc_dict


def fetch_prot_seq(pid, seq_only=True):
    """
    Fetch protein sequence from UniProt protal

    Args:
        pid: a valid protein UniProt ID
    Returns:
        organism name corresponds to the input protein
    """

    url = "https://rest.uniprot.org/uniprotkb/{pid}?format=json".format(pid=pid)

    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
        # info = re.findall('<name type="common">(.*?)</name>', content)
    js_dict = json.loads(content)
    if seq_only:
        return js_dict['sequence']['value']
    return js_dict


def get_prot_length(uprot_all, uprot2seq_dict):
    # uprot_all = var_df['UniProt'].drop_duplicates().values
    # resource_root = '/local/storage/yl986/data/uniprot_data_20220526/'
    # uprot2seq_dict = parse_fasta(os.path.join(resource_root, 'uniprot_sprot.fasta'))
    # isoform2seq_dict = parse_fasta(os.path.join(resource_root, 'uniprot_sprot_varsplic.fasta'))
    # uprot2seq_dict.update(isoform2seq_dict)
    uprot_all = set(uprot_all)
    print('Unique protein IDs: {}'.format(len(uprot_all)))
    uprot2length = dict()
    trembl = set()
    for uprot in uprot_all:
        try:
            seq = uprot2seq_dict[uprot]
        except KeyError:
            seq = fetch_prot_seq(uprot)
            trembl.add(uprot)
        uprot2length[uprot] = len(seq)

    return uprot2length

def aa_to_index(aa: str):
    """
    Encode amino acid to numerical values (prepare for one-hot encoding)
    """
    aa_index = {'ALA': 0,
                'CYS': 1,
                'ASP': 2,
                'GLU': 3,
                'PHE': 4,
                'GLY': 5,
                'HIS': 6,
                'ILE': 7,
                'LYS': 8,
                'LEU': 9,
                'MET': 10,
                'ASN': 11,
                'PRO': 12,
                'GLN': 13,
                'ARG': 14,
                'SER': 15,
                'THR': 16,
                'VAL': 17,
                'TRP': 18,
                'TYR': 19,
                'ASX': 20,
                'XAA': 20,
                'GLX': 20,
                'XLE': 20,
                'SEC': 20,
                'PYL': 20}

    return aa_index.get(aa.upper(), 20)


def calculate_expasy(seq, expasy_dict):
    """
    Calculate ExPaSy features (modified from PIONEER script)

    Args:
        seq (str): protein primary sequence
        expasy_dict (dict): ExPasy scales
    Returns:
        biochemical feature matrix (n_residues x 7)
    """

    # Calculate ExPaSy features
    feat_vec = []
    for feat in ['ACCE', 'AREA', 'BULK', 'COMP', 'HPHO', 'POLA', 'TRAN']:
        feat_vec.append(np.array([expasy_dict[feat][x] if x in expasy_dict[feat] else 0 for x in seq]))
    expasy_feat = np.column_stack(feat_vec)
    return expasy_feat


