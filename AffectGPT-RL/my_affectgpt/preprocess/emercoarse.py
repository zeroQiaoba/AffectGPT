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

#####################################################
## EMERCoarse Low-level Filter
#####################################################
# 1. filter out chinese characters
def func_filter_no_chinese(names, reasons):
    names_new, reasons_new = [], []
    for name, reason in zip(names, reasons):
        if text_has_chinese_char(reason):
            continue
        else:
            names_new.append(name)
            reasons_new.append(reason)
    return names_new, reasons_new

# 2. require reason to contain 'In the text'
def func_filter_no_inthetext(names, reasons):
    names_new, reasons_new = [], []
    for name, reason in zip(names, reasons):
        if reason.find('In the text') != -1:
            names_new.append(name)
            reasons_new.append(reason)
    return names_new, reasons_new

# 3. 'In the text' position must be within the gaussian range
def func_filter_wrong_pos(names, reasons):
    names_new, reasons_new = [], []
    for name, reason in zip(names, reasons):
        position = reason.find('In the text')/len(reason)
        if position > 0.6 and position < 0.8:
            names_new.append(name)
            reasons_new.append(reason)
    return names_new, reasons_new

# 4. description length distribution must also be within the gaussian range
def func_filter_prefilter(names, reasons, filternames):
    names_new, reasons_new = [], []
    for name, reason in tqdm.tqdm(zip(names, reasons)):
        if name in filternames:
            names_new.append(name)
            reasons_new.append(reason)
    return names_new, reasons_new

def EMERCorase_Filter():
    label_path = config.PATH_TO_LABEL['EMERCoarse']
    names = func_read_key_from_csv(label_path, 'name')
    reasons = func_read_key_from_csv(label_path, 'reason')
    print (len(reasons)) # 115595

    ## filter out reasons containing chinese
    names, reasons = func_filter_no_chinese(names, reasons)
    print (len(reasons))

    ## filter out reasons without 'In the text'
    names, reasons = func_filter_no_inthetext(names, reasons)
    print (len(reasons))
    
    ## filter out reasons whose 'In the text' position is wrong => it should be in [0.6, 0.8]
    names, reasons = func_filter_wrong_pos(names, reasons)
    print (len(reasons)) # 106969

    ## filter out wrong description length distribution
    filternames_path = os.path.join(config.DATA_DIR['EMERCoarse'], 'samplenames_filter_plus.npy')
    filternames = np.load(filternames_path)
    names, reasons = func_filter_prefilter(names, reasons, filternames)
    print (len(reasons)) # 106962

    ## store the results
    name2reason = {}
    for (name, reason) in zip(names, reasons):
        name2reason[name] = reason
    save_path = os.path.join(config.DATA_DIR['EMERCoarse'], 'merge_eng_filter.csv')
    func_write_key_to_csv(save_path, names, name2reason, ['reason']) 

# about 673/115595 errors: not found in the text
# minpos: 0.3804089396100808
# maxpos: 0.9951219512195122
# => the filter finally keeps 106962 samples

# 5. keep only the text content
def EMERCorase_Filter_text_only():
    input_csv = os.path.join(config.DATA_DIR['EMERCoarse'], 'merge_eng_filter.csv')
    names = func_read_key_from_csv(input_csv, 'name')
    reasons = func_read_key_from_csv(input_csv, 'reason')

    name2reason = {}
    for (name, reason) in tqdm.tqdm(zip(names, reasons)):
        assert reason.find('In the text') != -1
        reason = reason[reason.find('In the text'):]
        name2reason[name] = reason
    
    save_path = os.path.join(config.DATA_DIR['EMERCoarse'], 'merge_eng_filter_textonly.csv')
    func_write_key_to_csv(save_path, names, name2reason, ['reason']) 

    



###################################################################################
## EMERCoarse => valence / sentiment => valence range is [-1, 1]
###################################################################################
## 1. reason -> valence

def val_postprocess(valence_npz, save_npy):
    name2valnew = {}
    name2val = func_read_file(valence_npz)
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


## 2. read valence and post-process

## 3. show valence statistics [-1, 1] [ok]




############################################
## 1. ov -> sentiment

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

## 2. sent update
    
## 3. sent statistics




#####################################################
## EMERCoarse High-level Filter
#####################################################
## 1. PRED: get (ov, sent)
def generate_run_script():
    index = 0
    whole_run_templates = []
    for ii in range(10):
        names_npy = f'/data3/renyong/zeroqiaoba/dataset/EMER-Coarse/filter_split{ii}.npy'
        for outside_user_message in [
            "Please select the label that can best describe the person's emotional state from the provided candidate labels: angry,worried,sad,neutral,happy,surprise. ",
            "Please select the label that can best describe the person's emotional state from the provided candidate labels: fear,sadness,neutral,disgust,anger,surprise,joy.",
            "Please select the label that can best describe the person's emotional state from the provided candidate labels: neutral,anger,happy,sad.",
            "Please identify the overall positive or negative emotional polarity of the main characters. The output should be a ﬂoating-point number ranging from -3.0 to 3.0. Here, -3.0 indicates extremely negative emotions, 0 indicates neutral emotions, and 3.0 indicates extremely positive emotions. Please provide your judgment as a ﬂoating-point number.",
            "Please identify the overall positive or negative emotional polarity of the main characters. The output should be a ﬂoating-point number ranging from -1.0 to 1.0. Here, -1.0 indicates extremely negative emotions, 0 indicates neutral emotions, and 1.0 indicates extremely positive emotions. Please provide your judgment as a ﬂoating-point number. ",
                                     ]:
            run_template = f"""CUDA_VISIBLE_DEVICES=0 nohup python -u inference_hybird.py --dataset='EMERCoarseFilter' --process_names_npy=\"{names_npy}\" --outside_user_message=\"{outside_user_message}\" --cfg-path=train_configs/onlymerbench_outputhybird_bestsetup_bestfusion_largebatch_lz.yaml > lz_inference_bestfusion_largebatch{index}.out &"""
            index += 1
            whole_run_templates.append(run_template)
    
    for item in whole_run_templates:
        print (item)



## 2. PRED is split, so merge labels spread across multiple splits
def func_get_name2pred(result_path):

    name2pred = np.load(result_path, allow_pickle=True)['name2reason'].tolist()
    if len(name2pred) < 1000: # unfinished result_path
        return {}
    
    for name in name2pred:
        pred = name2pred[name]
        if pred.startswith('The most likely label is '):
            pred = pred[len('The most likely label is '):-1]
        elif pred.startswith("The character's emotional state is "):
            pred = pred[len("The character's emotional state is "):-1]
        elif pred.startswith("The valence score is "):
            pred = float(pred[len("The valence score is "):-1])
        else:
            ValueError('ERROR:', name, pred)
        name2pred[name] = pred
    return name2pred


def main_get_name2pred(result_root, debug=False):
    name2whole = {}
    for subroot in glob.glob(result_root + '/*'):
        for filepath in glob.glob(subroot + '/*'):
            name2pred = func_get_name2pred(filepath)
            for name in name2pred:
                if name not in name2whole:
                    name2whole[name] = []
                name2whole[name].append(name2pred[name])
    
    # debug: keep only 100 samples for testing
    if debug:
        newmap = {}
        for name in name2whole:
            if len(name2whole[name]) >= 4:
                newmap[name] = name2whole[name]
            if len(newmap) == 100:
                return newmap
    else:
        return name2whole
    


## 3. discrete metrics
# (pred_ov, pred_sent) -> pred_ov
def func_preds_to_ov(pred_candidates):
    pred_ovs = []

    for pred in pred_candidates:
        # for pred_ov process
        if isinstance(pred, str): 
            pred_ovs.append(pred)
        # for valence process
        else:
            if pred > 0:
                pred_sent = 'positive'
            elif pred < 0:
                pred_sent = 'negative'
            else:
                pred_sent = 'neutral'
            pred_ovs.append(pred_sent)
    return list(set(pred_ovs))

# case1: gt_ov and pred_ov hit if any candidate overlaps
def metric1_discrete_hit_or_not(gt_ov, pred_candidates):
    pred_ovs = func_preds_to_ov(pred_candidates)
    if hit_or_not(gt_ov, pred_ovs):
        return True
    else:
        return False
    

## 4. sentiment metrics
# (pred_ov, pred_sent) -> pred_sent
def func_preds_to_sent(pred_candidates):
    dis2sent = {'neutral':  ['neutral'],
                'happy':    ['positive'],
                'joy':      ['positive'],
                'angry':    ['negative'],
                'worried':  ['negative'],
                'sad':      ['negative'],
                'sadness':  ['negative'],
                'fear':     ['negative'],
                'anger':    ['negative'],
                'disgust':  ['negative'],
                'scared':   ['negative'],
                'worry':    ['negative'],
                'positive': ['positive'],
                'negative': ['negative'],
                'formal':   ['neutral'],
                'nervous':  ['negative', 'neutral'],
                'determined to be neutral': ['neutral'],
                'repeat':   ['neutral'],
                'humbled':  ['neutral'],
                'lonely':   ['negative'],
                'promise':  ['neutral'],
                'shame':    ['positive', 'neutral'],
                'respectful': ['positive'],
                'disgusted': ['negative'],
                'confident': ['positive', 'neutral'],
                'unhappy':   ['negative'],
                'sorry':     ['negative'],
                'proud':     ['positive'],
                'touched':  ['positive', 'negative', 'mix'],
                'surprise': ['positive', 'negative', 'mix']}

    pred_sents = []
    for pred in pred_candidates:
        # for pred_ov process
        if isinstance(pred, str) and pred in dis2sent: 
            pred_sent = dis2sent[pred]
        # for valence process
        elif isinstance(pred, float):
            if pred > 0:
                pred_sent = ['positive']
            elif pred < 0:
                pred_sent = ['negative']
            else:
                pred_sent = ['neutral']
        else:
            continue
        pred_sents.extend(pred_sent)
    return pred_sents


# case3: gt_sent must appear in pred_sent
def metric3_sent_hit_or_not(gt_sent, pred_candidates):
    assert gt_sent in ['neutral', 'positive', 'negative', 'mix']
    pred_sents = func_preds_to_sent(pred_candidates)

    '''
    gt_sent: positive
    pred_sents: ['neutral', 'neutral', 'positive']
    => return True
    '''
    if gt_sent in pred_sents:
        return True
    else:
        return False

# case4: major(pred_sent) == gt_sent; stricter than case3
def metric4_sent_hit_or_not(gt_sent, pred_candidates):
    assert gt_sent in ['neutral', 'positive', 'negative', 'mix']
    pred_sents = func_preds_to_sent(pred_candidates)

    '''
    gt_sent: positive
    pred_sents: ['neutral', 'neutral', 'positive']
    => return False
    '''
    if gt_sent in func_major_labels(pred_sents):
        return True
    else:
        return False


def func_filter(metric, process_names, gt_name2ov, gt_name2sent, name2preds):
    
    assert metric in ['metric1', 'metric3', 'metric4']

    filter_names = []
    for name in tqdm.tqdm(process_names):
        if metric == 'metric1':
            if metric1_discrete_hit_or_not(gt_name2ov[name], name2preds[name]):
                filter_names.append(name)
        elif metric == 'metric3':
            if metric3_sent_hit_or_not(gt_name2sent[name], name2preds[name]):
                filter_names.append(name)
        elif metric == 'metric4':
            if metric4_sent_hit_or_not(gt_name2sent[name], name2preds[name]):
                filter_names.append(name)
    return filter_names


def main_filter(label_csv, metrics, debug=False):
    
    # 1. read preds
    name2preds = main_get_name2pred('output/results-emercoarsefilter', debug)
    print (f'pred sample number: {len(name2preds)}') # 106962
    assert len(name2preds) == 106962


    # 2. read gt
    ov_label   = os.path.join(config.DATA_DIR['EMERCoarse'], 'ov_labels_textonly.npz')
    sent_label = os.path.join(config.DATA_DIR['EMERCoarse'], 'sent_textonly_new.npy')
    gt_name2ov = func_read_file(ov_label)
    gt_name2sent = np.load(sent_label, allow_pickle=True).tolist()
    assert len(gt_name2ov) == 106962 and len(gt_name2sent) == 106962

    # 3. filter and save
    filter_names = list(name2preds.keys())
    print (f'whole samples: {len(filter_names)}')
    for metric in metrics.split('_'):
        filter_names = func_filter(metric, filter_names, gt_name2ov, gt_name2sent, name2preds)
    print (f'filter samples: {len(filter_names)}')

    # 4. store the results
    name2reason_filter = {}
    names = func_read_key_from_csv(label_csv, 'name')
    reasons = func_read_key_from_csv(label_csv, 'reason')
    for (name, reason) in zip(names, reasons):
        if name in filter_names:
            name2reason_filter[name] = reason
    keynames = ['reason']
    save_csv = label_csv[:-4] + f'_{metrics}.csv'
    func_write_key_to_csv(save_csv, filter_names, name2reason_filter, keynames)
    




def filter_speaker_or_not(label_path, save_csv):

    print ('============================================================')
    name_mapping_path = os.path.join(config.DATA_DIR['EMERCoarse'], "old2new-release.npz")
    name_mapping = np.load(name_mapping_path, allow_pickle=True)['old2new'].tolist()

    # read speaker info
    name2spk = {}
    speaker_info_path = os.path.join(config.DATA_DIR['EMERCoarse'], "speak_or_not.csv")
    names = func_read_key_from_csv(speaker_info_path, 'filename')
    infos = func_read_key_from_csv(speaker_info_path, 'result')
    for (name, info) in zip(names, infos):
        if name in name_mapping:
            name = name_mapping[name]
        name2spk[name] = info
    
    # read label_path
    name2reason_filter = {}
    names = func_read_key_from_csv(label_path, 'name')
    reasons = func_read_key_from_csv(label_path, 'reason')
    print (f'original sample number: {len(names)}')
    for (name, reason) in zip(names, reasons):
        if name2spk[name] == 1:
            name2reason_filter[name] = reason

    # save to .csv file
    filter_names = [name for name in name2reason_filter]
    print (f'filter sample number: {len(filter_names)}')
    func_write_key_to_csv(save_csv, filter_names, name2reason_filter, ['reason'])
    print ('============================================================')


# keep only the (audio, text) match part


############################
## OV label distribution statistics
############################
def calculate_statistic_for_ov_emerfine(label_path, train_test_split, type='train'):

    name2openset = {}
    names = func_read_key_from_csv(label_path, 'name')
    opensets = func_read_key_from_csv(label_path, 'openset')
    for (name, openset) in zip(names, opensets):
        name2openset[name] = openset

    # => process_names
    train_names = list(np.load(train_test_split, allow_pickle=True)['train_names'])
    test_names  = list(np.load(train_test_split, allow_pickle=True)['test_names'])
    if type == 'train':
        process_names = train_names
    elif type == 'whole':
        process_names = train_names + test_names
    print (f'sample number: {len(process_names)}') # 54326

    whole_ovs = []
    for name in process_names:
        whole_ovs.extend(string_to_list(name2openset[name]))
    print (f'unique label: {len(set(whole_ovs))}') # 1074

    # count labels appearing more than 6 times
    label_filter = []
    label2count = func_label_distribution(whole_ovs)
    for label in label2count:
        if label2count[label] >= 6:
            label_filter.append(label)
    print (f'at least appear 6 times: {len(set(label_filter))}') # 282



def calculate_statistic_for_ov_emercoarse(openset_path):

    fileitems = np.load(openset_path, allow_pickle=True)['fileitems']
    print (f'sample number: {len(fileitems)}')

    whole_ovs = []
    for item in tqdm.tqdm(fileitems):
        whole_ovs.extend(string_to_list(item))
    print (f'unique label: {len(set(whole_ovs))}')

    # count labels appearing more than 6 times
    label_filter = []
    label2count = func_label_distribution(whole_ovs)
    for label in label2count:
        if label2count[label] >= 6:
            label_filter.append(label)
    print (f'at least appear 6 times: {len(set(label_filter))}')




############################
## Video Duration distribution
############################
def calculate_duration():
    audio_root = config.PATH_TO_RAW_AUDIO['EMERCoarse']
    save_path = os.path.join(config.DATA_DIR['EMERCoarse'], 'duration.npy')

    name2duration = {}
    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)[:-4]
        duration = func_read_audio_second(audio_path)
        name2duration[name] = duration
    np.save(save_path, name2duration)



def draw_duration_lens(save_png, binnum=60, color='b'):
    
    save_path = os.path.join(config.DATA_DIR['EMERCoarse'], 'duration.npy')
    name2duration = np.load(save_path, allow_pickle=True).tolist()
    data = [name2duration[name] for name in name2duration]
    print (f'sample number: {len(data)}')

    plt.figure()

    # => plot histogram
    min_val = min(data)
    max_val = max(data)
    iterval = (max_val - min_val) / binnum
    bins = np.arange(min_val, max_val+iterval, iterval)
    plt.hist(data, bins, density=True, color=color)
    
    fontsize = 16
    plt.xlabel('Duration (s)',  fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.xlim(0, 20)
    #plt.ylim(0, 0.04)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_png, format='png')
    plt.close()

    

