import os
import cv2
import time
import copy
import glob
import tqdm
import json
import openai
import shutil
import random
import pathlib
import argparse
import numpy as np
import pandas as pd
import multiprocessing

import config

from toolkit.utils.functions import *
from toolkit.utils.read_files import *

from my_affectgpt.evaluation.ew_metric import *
from my_affectgpt.evaluation.wheel import *


def func_read_file(label_path):
    name2item = {}
    filenames = np.load(label_path, allow_pickle=True)['filenames']
    fileitems = np.load(label_path, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2item[name] = item
    return name2item


###################################################################################
## 1. split the good answer out of the preference data
###################################################################################
def preference_good_answer_split(label_csv, good_csv):
    name2reason = {}
    df = pd.read_csv(label_csv)
    for _, row in df.iterrows():
        name = row['name']
        a1 = row['a1']
        a2 = row['a2']
        preference = row['preference']
        if preference == 'a1':
            name2reason[name] = a1
        elif preference == 'a2':
            name2reason[name] = a2
    
    ## write to csv
    names = list(name2reason.keys())
    keynames = ['reason']
    func_write_key_to_csv(good_csv, names, name2reason, keynames)


###################################################################################
## 2. good -> ov labels
###################################################################################
def ov_extraction(good_csv, ov_npz):
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    extract_openset_batchcalling(reason_csv=good_csv, store_npz=ov_npz,
                                llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)



###################################################################################
## 3. good -> valence
###################################################################################
def valence_extraction(good_csv, val_npz):
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    reason_to_valence_batchcalling(label_csv=good_csv, store_npz=val_npz,
                                llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

def func_val_postprocess(val_npz, save_npy):
    name2valnew = {}
    name2val = func_read_file(val_npz)
    for name in tqdm.tqdm(name2val):
        val = name2val[name]
        
        if val.find('would be:') != -1:
            val = val[val.find('would be:'):]
        elif val.find('Output:') != -1:
            val = val[val.find('Output:'):]
        
        match = re.search(r"[-+]?\d*\.?\d+", val)
        assert match, f'must have matched values'
        val_new = float(match.group(0))

        name2valnew[name] = val_new
    np.save(save_npy, name2valnew)


############################################
## 3. good -> sentiment
############################################
## 1. ov -> sentiment
def sentiment_extraction(ov_npz, sent_npz):
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    openset_to_sentiment_batchcalling(openset_npz=ov_npz, store_npz=sent_npz, 
                                    llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

## 2. sent update
def sentiment_postprocess(sent_npz, save_npy):
    name2sentnew = {}
    name2sent = func_read_file(sent_npz)
    for name in tqdm.tqdm(name2sent):
        sent = name2sent[name]

        # gt sent post-process => (gt_new)
        sent_new = ""
        if sent.find('negative') != -1:
            sent_new = 'negative'
        elif sent.find('positive') != -1 or sent.find('playful') != -1:
            sent_new = 'positive'
        elif sent.find('neutral') != -1:
            sent_new = 'neutral'
        elif sent.find('mix') != -1 or sent.find('complex') != -1 or sent.find('ambiguous') != -1:
            sent_new = 'mix'
        assert sent_new != ''

        name2sentnew[name] = sent_new
    np.save(save_npy, name2sentnew)


###################################################################################
## Global path setting
###################################################################################
## case1: experiment on the preference data labeled by three people


## case2: train an SFT model on a small human-labeled preference set to label the other (pair) data
for augid in range(1, 7):
    good_csv   = os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good.csv')
    ov_npz     = os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good_ov.npz')
    val_npz    = os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good_valence.npz')
    valnew_npy = os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good_valence_new.npy')
    sent_npz   = os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good_sent.npz')
    sentnew_npy= os.path.join(config.DATA_DIR['Preference'], f'preference_aug{augid}_good_sent_new.npy')
    ov_extraction(good_csv, ov_npz)
    # (name2reason) => (name2val)
    valence_extraction(good_csv, val_npz)
    func_val_postprocess(val_npz, valnew_npy)
    name2val = np.load(valnew_npy, allow_pickle=True).tolist()
    minval = min([name2val[name] for name in name2val])
    maxval = max([name2val[name] for name in name2val])
    print (minval, maxval)
    # (name2reason) => (name2sent)
    sentiment_extraction(ov_npz, sent_npz)
    sentiment_postprocess(sent_npz, sentnew_npy)
    name2sent = np.load(sentnew_npy, allow_pickle=True).tolist()
    candidates = set([name2sent[name] for name in name2sent])
    print (candidates)
