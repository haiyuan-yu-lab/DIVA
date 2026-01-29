import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from pathlib import Path
import yaml
import json
import pickle
import copy
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM
from transformers import BertConfig, EsmConfig

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator, PhenotypeDataset, TextDataCollator
import logging
from datetime import datetime
from utils import str2bool, _save_scores
from metrics import *
from dev.preprocess.utils import parse_fasta_info
from dev.disease_inference import load_config, env_setup, embed_phenotypes, inference
from models.dis_var_models import DiseaseVariantEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/bin_var_pred_config.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency to save models')
    # parser.add_argument('--inf-check', type=str2bool, default=False,
    #                     help='add hooks to check for infinite module outputs and gradients')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)
    config, device = env_setup(args, config)
    data_configs = config['dataset']
    model_args = config['model']
    exp_path = Path(config['exp_dir'])

    prot2seq = dict()
    prot2desc = dict()
    data_root = Path(data_configs['data_dir'])
    for fname in data_configs['seq_fasta']:
        try:
            seq_dict, desc_dict = parse_fasta_info(fname)
            prot2seq.update(seq_dict)
            prot2desc.update(desc_dict)  # string of protein definition E.g. BRCA1_HUMAN Breast cancer type 1 susceptibility protein
        except FileNotFoundError:
            pass

    for fpath in data_configs['prot_meta_data']:
        try:
            df_meta = pd.read_csv(fpath, sep='\t').dropna(subset=['function'])
            # meta_info_all.append(df)
            prot_func_dict = dict(zip(df_meta['UniProt'], df_meta['function']))
            prot2desc.update(prot_func_dict)
        except FileNotFoundError:
            pass
        
    for prot, desc in prot2desc.items():
        # update description of isoforms
        if prot.find('-') >= 0:
            prot2desc[prot] = ' '.join([desc, prot2desc.get(prot.split('-')[0], '')]).strip()

    prot2comb_seq = None
    data_configs['seq_dict'] = prot2seq
    data_configs['protein_info_dict'] = prot2desc

    afmis_root = None
    if data_configs['use_alphamissense']:
        afmis_root = Path(data_configs['alphamissense_score_dir'])

    # Initialize tokenizer
    protein_tokenizer = AutoTokenizer.from_pretrained(model_args['protein_lm_path'],
        do_lower_case=False
    )
    text_tokenizer = BertTokenizer.from_pretrained(model_args['text_lm_path'])

    # Load data
    with open(data_configs['phenotype_vocab_file'], 'r') as f:
        phenotype_vocab = f.read().splitlines()
    phenotype_vocab.insert(0, text_tokenizer.unk_token)  # add unknown token
    logging.info('Disease vocabulary size: {}'.format(len(phenotype_vocab)))
    if data_configs['use_pheno_desc']:
        with open(data_configs['phenotype_desc_file']) as f:
            pheno_desc_dict = json.load(f)
            logging.info('Disease description file loaded.')
    else:
        pheno_desc_dict = None
    if data_configs.get('disease_name_map_file', None):  # raw disease names --> cleaned terms
        with open(data_configs['disease_name_map_file']) as f:
            dis_name_map_dict = json.load(f)
        logging.info('Disease name mapping file loaded.')
    else:
        dis_name_map_dict = None

    pheno_dataset = PhenotypeDataset(phenotype_vocab, pheno_desc_dict, use_desc=data_configs['use_pheno_desc'])
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    
    seq_config = BertConfig.from_pretrained(model_args['protein_lm_path'])
    text_config = BertConfig.from_pretrained(model_args['text_lm_path'])

    seq_encoder = EsmForMaskedLM(seq_config)
    text_encoder = BertForMaskedLM(text_config)
    
    model = DiseaseVariantEncoder(seq_encoder=seq_encoder,
                                  text_encoder=text_encoder,
                                  n_residue_types=protein_tokenizer.vocab_size,
                                  hidden_size=512,
                                  use_desc=True,
                                  pad_label_idx=-100,
                                  calibration_fn_name=model_args['calibration_fn_name'],
                                  dist_fn_name=model_args['dist_fn_name'],
                                  init_margin=model_args['margin'],
                                  use_alphamissense=data_configs['use_alphamissense'],
                                  adjust_logits=model_args['adjust_logits'],
                                  device=device)
    checkpt_dict = torch.load(config['model_path'], map_location='cpu')
    model.load_state_dict(checkpt_dict['state_dict'], strict=False)

    for name, parameters in model.named_parameters():
        parameters.requires_grad = False
    
    model = model.to(device)
    all_pheno_embs = embed_phenotypes(model, device, phenotype_loader)
    all_pheno_embs = torch.tensor(all_pheno_embs, device=device)
    if isinstance(data_configs['input_file']['test'], str):
        test_flist = [data_configs['input_file']['test']]
    else:
        test_flist = data_configs['input_file']['test']

    for test_file in test_flist:
        logging.info(f'Inference on {test_file}...')
        fname = os.path.basename(test_file).split('.')[0]
        test_dataset = ProteinVariantDatset(**data_configs, 
                                            variant_file=test_file, 
                                            split='test', 
                                            phenotype_vocab=phenotype_vocab, 
                                            protein_tokenizer=protein_tokenizer, 
                                            text_tokenizer=text_tokenizer,
                                            #  var_db=var_db,
                                            # prot_var_cache=prot_var_cache,
                                            mode='eval',
                                            update_var_cache=False,
                                            comb_seq_dict=prot2comb_seq,
                                            # disease_name_map_dict=dis_name_map_dict,
                                            afmis_root=afmis_root,
                                            access_to_context=False)
        logging.info('{} variants loaded ({} with known disease label)'.format(len(test_dataset), test_dataset.n_disease_variants))
        test_collator = ProteinVariantDataCollator(test_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                               use_prot_desc=True, truncate_protein=data_configs['truncate_protein'], 
                                               max_protein_length=data_configs['max_protein_seq_length'],
                                               use_alphamissense=data_configs['use_alphamissense'],
                                               use_pheno_desc=data_configs['use_pheno_desc'], pheno_desc_dict=pheno_desc_dict)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=test_collator)
        
        test_labels, test_scores, test_vars, test_adj_weights, test_pheno_results = inference(model, device, test_loader, pheno_vocab_emb=all_pheno_embs, topk=100)

        _save_scores(test_vars, test_labels, test_scores, fname, weights=test_adj_weights, epoch='', exp_dir=str(exp_path), mode='eval')
        logging.info('Done!')

    # np.save(exp_path / 'phenotype_emb.npy', all_pheno_embs.detach().cpu().numpy())
