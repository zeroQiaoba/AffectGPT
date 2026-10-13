import os
import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from my_affectgpt.datasets.builders.image_text_pair_builder import * # pull in every dataset class
from my_affectgpt.evaluation.ew_metric import *

############################################################################
## Functions
############################################################################
# 1. search for the matching model file
def search_for_result_root(input_dir, inter_print=True):
    candidates = glob.glob(input_dir + '*')
    root_candidates = [root for root in candidates if os.path.isdir(root)]
    if len(root_candidates) == 0:
        if inter_print: print ('No file exists!')
        return ''

    # keep the root that holds the most result files
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        store_path = []
        for path in os.listdir(root):
            if path.startswith('checkpoint_') and path.find('-') == -1:
                store_path.append(path)
        count = len(store_path)
        if inter_print: print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root

    if inter_print: print ('================================================')
    if inter_print: print ('Targetroot: ', targetroot)
    if inter_print: print ('Saved result files ', maxcount)
    # report last file info
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    if inter_print: print("Last result file creation time:", datetime.datetime.fromtimestamp(creation_time))
    if inter_print: print ('================================================')
    return targetroot


# 2. resolve the dataset name
def func_read_datasetname(input_dir=None, dataset=None):

    # => (dataset)
    if input_dir is not None:
        assert input_dir.find('/results-') != -1
        dataset = input_dir.split('/results-')[1].split('/')[0]
    else:
        assert dataset is not None

    # (dataset) => (supprot_item)
    supprot_datasets = list(config.DATA_DIR.keys())
    for supprot_item in supprot_datasets:
        if supprot_item.lower() == dataset.lower():
            return supprot_item

    ValueError(f'cannot find suitable dataset for {input_dir}')


# 3. load the dataset
def get_dataset2cls(dataset):
    if dataset == 'OVMERDPlus':  return OVMERDPlus_Dataset()
    print ('dataset cls not provided!')
    return None


# load vLLM models
def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


############################################################################
## metric calculation on OV-MERD+: reason -> openset -> EW-based metric
############################################################################
def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):

    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # compute the EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    fscore, precision, recall = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return fscore, precision, recall


## main function: score every epoch stored under one ckpt_root
def main_zeroshot_scores(input_dir, debug=False, inter_print=True,
                         llm=None, tokenizer=None, sampling_params=None):

    # when input_dir does not exist yet, resolve it to the best matching path
    if not os.path.exists(input_dir):
        input_dir = search_for_result_root(input_dir, inter_print)
    if inter_print: print (f'process root: {input_dir}')

    # read dataset infos
    dataset = func_read_datasetname(input_dir)
    assert dataset == 'OVMERDPlus', f'run-release.sh only infers on OV-MERD+, but got {dataset}'
    if inter_print: print (f'process dataset: {dataset}')
    dataset_cls = get_dataset2cls(dataset)
    name2gt = dataset_cls.get_test_name2gt()
    if inter_print: print (f'target sample number: {len(name2gt)}')

    # load model
    if (debug==False) and (llm is None) and (tokenizer is None) and (sampling_params is None):
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')

    # main process
    epoch2result = {}
    for epoch_root in sorted(glob.glob(input_dir + '/*.npz')):
        epoch_name = os.path.basename(epoch_root)[:-4]

        if epoch_name.find('openset') != -1 or epoch_name.find('sentiment') != -1:
            continue

        # =============== process for {epoch_root} ===============
        if epoch_name.find('_') == -1: # epoch_name is not of the expected form
            epoch = 0
            reward = 0.00
            epoch2result[epoch] = {'reward': reward}
        else: # epoch_name is of the expected form
            epoch = epoch_name.split('_')[1]
            if epoch_name.find('_reward_') != -1:
                reward = float(epoch_name.split('_')[3])
            else:
                reward = 0.00
            epoch2result[epoch] = {'reward': reward}

        # 1. score calculation
        fscore, _, _ = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
        epoch2result[epoch]['score'] = fscore

    return epoch2result
    # store two values: {'reward': xxx, 'score': xxx}


###############################################################
## reporting: the two stages report differently, so both switches are kept
###############################################################
##   show_epoch=<int>      -> print the score of that single epoch (Stage 1 reports the convergence point, epoch 60)
##   show_maxscores_0_5_60 -> print the best score over every epoch sampled with a 5-epoch step (Stage 2)
def main_show_statistic_ovlabel(model_name, output_folder='output',
                                show_maxscores_0_5_60=True, show_epoch=None):

    process_root = f"{output_folder}/results-ovmerdplus/{model_name}"
    epoch2score = main_zeroshot_scores(process_root, debug=True, inter_print=False)

    if show_epoch is not None:
        ## the epoch key comes from the npz filename: checkpoint_000060_loss_0.294.npz -> '000060'
        ## it is zero-padded, so compare as int rather than as string
        store_scores = [epoch2score[epoch]['score'] * 100 for epoch in epoch2score
                        if epoch.isdigit() and int(epoch) == int(show_epoch)]
        if len(store_scores) == 0:
            print (f'epoch:{show_epoch}; score: not found')
        else:
            print (f'epoch:{int(show_epoch)}; score: {max(store_scores):.2f}')

    if show_maxscores_0_5_60: # best score over every epoch sampled with a 5-epoch step
        store_scores = []
        for epoch in sorted(epoch2score):
            if int(epoch)%5 == 0:
                store_scores.append(epoch2score[epoch]['score'] * 100)
        print (f'score: {max(store_scores):.2f}')


# run on one GPU: CUDA_VISIBLE_DEVICES=3 python statistic.py
if __name__ == "__main__":

    ## run-release.sh runs inference only on OV-MERD+, once after Stage-1 and once after Stage-2.
    ## Both write to output/results-ovmerdplus/<ckpt_root>/, and ckpt_root carries a timestamp
    ## assigned at training time -- so the two entries below are prefixes only, and
    ## search_for_result_root() resolves each of them to the actual result dir.
    ##
    ## the third field is how that run is reported, and the two stages differ:
    ##   stage1: training stops at max_epoch=60, so report the score of the convergence point (epoch 60);
    ##   stage2: continues from the stage1 ckpt for 60 more epochs, so report the best score sampled every 5.
    candidate_infos = [
        # Stage 1 (cold start): mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage1
        ("mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage1", "output",
         {'show_maxscores_0_5_60': False, 'show_epoch': 60}),
        # Stage 2 (RL): mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage2_mer25ov1000_continue60_g4_formatbeta00_ovlabel_numrestrict3
        ("mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage2_mer25ov1000_continue60_g4_formatbeta00_ovlabel_numrestrict3", "output",
         {'show_maxscores_0_5_60': True}),
    ]

    ################################################################
    ########## inference on OV-MERD+ (the only dataset this release reports on) ##########
    ################################################################

    ## step1: ov label extraction
    for modelname, outputfolder, _ in candidate_infos:
        print ("================================================")
        print (modelname)
        main_zeroshot_scores(f"{outputfolder}/results-ovmerdplus/{modelname}")

    ## step2: score statistic
    for modelname, outputfolder, show_cfg in candidate_infos:
        print ("================================================")
        print (modelname)
        main_show_statistic_ovlabel(model_name=modelname, output_folder=outputfolder, **show_cfg)
