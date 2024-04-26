import re
import os
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

###########################################
# common function
###########################################
import openai
def get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print ('Cannot connect to abroad network.')
        return ''

def func_gain_name2value(csv_path, value='reasons'):
    name2value = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        name = row['names']
        reason = row[value]
        if pd.isna(reason): reason=""
        name2value[name] = reason
    return name2value


###########################################
# evaluate clue overlap
###########################################
def func_clue_scoring(gt_reason, pred_reason):
    prompt = f"""
                下面给出人物的《真实描述》以及《预测描述》。请按照以下步骤计算《预测描述》的得分，得分范围为1-10。最终仅输出预测分数的数值大小，并给出原因。

                1.请总结《真实描述》中有关人物的情感状态描述

                2.请总结《预测描述》中有关人物的情感状态描述

                3.计算《预测描述》与《真实描述》之间的重叠度，重叠度越高，返回的分数越高。

                4.输出格式为: '预测分数'：预测分数；'原因'：原因
                
                输入：

                《真实描述》：{gt_reason}

                《预测描述》：{pred_reason}

                输出：

                """
    response = get_completion(prompt)
    return response

def chatgpt_clue_scoring_main(data_root, save_path, debug=False):

    ## read ground-truth label
    gt_path = os.path.join(data_root, 'gt.csv')
    name2gt = func_gain_name2value(gt_path, value='chi_reasons')

    ## read predicted labels
    name2preds = {}
    pred_paths = [os.path.join(data_root, 'pandagpt.csv'),
                  os.path.join(data_root, 'valley.csv'),
                  os.path.join(data_root, 'videochat(emb).csv'),
                  os.path.join(data_root, 'videochat(text).csv'),
                  os.path.join(data_root, 'videochatgpt.csv'),
                  os.path.join(data_root, 'videollama.csv')]
    for pred_path in pred_paths:
        predname = os.path.basename(pred_path)[:-4]
        name2reason = func_gain_name2value(pred_path, value='chi_reasons')
        for ii, name in enumerate(name2reason):
            if debug and ii == 2: break
            if name not in name2preds:
                name2preds[name] = []
            name2preds[name].append((predname, name2reason[name]))
    
    # save chatgpt_score
    name2score = {}
    for name in name2preds:
        print (f'====== {name} ======')
        gt = name2gt[name]
        for (predname, predreason) in tqdm.tqdm(name2preds[name]):
            score = func_clue_scoring(gt, predreason)
            start, end = re.search(r"[\d][.]*[\d]*", score).span()
            score = float(score[start:end])
            if name not in name2score: name2score[name] = []
            print (score, predreason)
            name2score[name].append((predname, score))
    np.savez_compressed(save_path, name2score=name2score)


###########################################
# evaluate emotion overlap
###########################################
def get_summarized_emotions(reason):
    prompt = f"""
              请根据以下视频描述，推测视频中人物最有可能的情感状态，仅输出情感词：
              
              视频描述：他虽然看起来很高兴，但是实际上很焦虑
              
              输出结果：焦虑

              视频描述：{reason}

              输出结果：
              """
    response = get_completion(prompt)
    return response

def func_label_scoring(gt_reason, pred_reason):
    prompt = f"""
                下面给出人物的《真实情感》以及《预测情感》。请计算《预测情感》与《真实情感》之间的重叠度，重叠度越高，返回的分数越高。得分范围为1-10。最终仅输出预测分数的数值大小，并给出原因。
                
                输出格式为: '预测分数'：预测分数；'原因'：原因
                
                输入：

                《真实情感》：{gt_reason}

                《预测情感》：{pred_reason}

                输出：

                """
    response = get_completion(prompt)
    return response

def chatgpt_label_scoring_main(data_root, save_path, debug=False):

    ## read ground-truth label
    gt_path = os.path.join(data_root, 'gt.csv')
    name2gt = func_gain_name2value(gt_path, value='chi_reasons')

    ## read predicted labels
    name2preds = {}
    pred_paths = [os.path.join(data_root, 'pandagpt.csv'),
                  os.path.join(data_root, 'valley.csv'),
                  os.path.join(data_root, 'videochat(emb).csv'),
                  os.path.join(data_root, 'videochat(text).csv'),
                  os.path.join(data_root, 'videochatgpt.csv'),
                  os.path.join(data_root, 'videollama.csv')]
    for pred_path in pred_paths:
        predname = os.path.basename(pred_path)
        predname = predname.split('_')[-1][:-4]
        name2reason = func_gain_name2value(pred_path, value='chi_reasons')
        for ii, name in enumerate(name2reason):
            if debug and ii == 2: break
            if name not in name2preds:
                name2preds[name] = []
            name2preds[name].append((predname, name2reason[name]))
            
    ## save chatgpt_score
    name2score = {}
    for name in name2preds:
        print (f'====== {name} ======')
        for (predname, predreason) in tqdm.tqdm(name2preds[name]):
            gt_emotion = get_summarized_emotions(name2gt[name])
            pred_emotion = get_summarized_emotions(predreason)
            score = func_label_scoring(gt_emotion, pred_emotion)
            start, end = re.search(r"[\d][.]*[\d]*", score).span()
            score = float(score[start:end])
            if name not in name2score: name2score[name] = []
            print (score, predreason)
            name2score[name].append((predname, score))
    np.savez_compressed(save_path, name2score=name2score)


###########################################
# analyze scores for different baselines
###########################################
# select_names: list file contains several names. Here, select_names=[] represents to process on all names
def chatgpt_scoring_analyze(score_path, select_names=[], print_flag=True):
    whole_score = {}
    name2score = np.load(score_path, allow_pickle=True)['name2score'].tolist()
    for name in name2score:
        if len(select_names)!=0 and (name not in select_names):
            continue
        for (modelname, score) in name2score[name]:
            if modelname not in whole_score:
                whole_score[modelname] = []
            whole_score[modelname].append(score)
    
    for modelname in whole_score:
        scores = whole_score[modelname]
        if print_flag:
            print (f'processed sample numbers: {len(scores)}')
        meanscore = np.mean(scores)
        print (f'{modelname} == average score: {meanscore}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str, default=None, help='dataset path')
    parser.add_argument('--openai_key', type=str, default=None, help='you chatgpt key')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    args = parser.parse_args()
    openai.api_key = args.openai_key

    ## for clue overlap analysis
    result_path = os.path.join(args.data_root, 'clue_score.npz')
    chatgpt_clue_scoring_main(args.data_root, result_path, args.debug)
    chatgpt_scoring_analyze(result_path)

    ## for label overlap analysis
    result_path = os.path.join(args.data_root, 'emo_score.npz')
    chatgpt_label_scoring_main(args.data_root, result_path, args.debug)
    chatgpt_scoring_analyze(result_path)