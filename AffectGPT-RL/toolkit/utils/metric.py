import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *

# 综合维度和离散的评价指标
def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score


# 只返回 metric 值，用于模型筛选 
def gain_metric_from_results(eval_results, metric_name='emoval'):

    if metric_name == 'emoval':
        fscore = eval_results['emofscore']
        valmse = eval_results['valmse']
        overall = overall_metric(fscore, valmse)
        sort_metric = overall
    elif metric_name == 'emo':
        fscore = eval_results['emofscore']
        sort_metric = fscore
    elif metric_name == 'val':
        valmse = eval_results['valmse']
        sort_metric = -valmse
    elif metric_name == 'loss':
        loss = eval_results['loss']
        sort_metric = -loss

    return sort_metric


def gain_cv_results(folder_save):

    # find all keys
    whole_keys = list(folder_save[0].keys())

    cv_acc, cv_fscore, cv_valmse = -100, -100, -100
    if 'eval_emoacc' in whole_keys:
        cv_acc = np.mean([epoch_save['eval_emoacc'] for epoch_save in folder_save])
    if 'eval_emofscore' in whole_keys:
        cv_fscore = np.mean([epoch_save['eval_emofscore'] for epoch_save in folder_save])
    if 'eval_valmse' in whole_keys:
        cv_valmse = np.mean([epoch_save['eval_valmse'] for epoch_save in folder_save])
    
    # 只显示存在的部分信息 [与test输出是一致的]
    outputs = []
    if cv_fscore != -100: outputs.append(f'f1:{cv_fscore:.4f}')
    if cv_acc    != -100: outputs.append(f'acc:{cv_acc:.4f}')
    if cv_valmse != -100: outputs.append(f'val:{cv_valmse:.4f}')
    outputs = "_".join(outputs)
    return outputs


def average_folder_for_emos(folder_save, testname):

    try:
        # 因为所有test set的 shuffle都是false的，因此不同folder的结果是对应的
        labels = folder_save[0][f'{testname}_emolabels']
    except:
        return [], []

    num_samples = len(labels)
    num_folders = len(folder_save)

    whole_probs = []
    for ii in range(num_folders):
        emoprobs = folder_save[ii][f'{testname}_emoprobs']
        whole_probs.append(emoprobs)
    whole_probs = np.array(whole_probs)

    avg_preds = []
    for ii in range(num_samples):
        per_probs = whole_probs[:, ii, :]
        avg_emoprob = np.mean(per_probs, axis=0)
        avg_preds.append(avg_emoprob)
    
    return labels, avg_preds

# 计算 name -> val
def average_folder_for_vals(folder_save, testname):

    try:
        labels = folder_save[0][f'{testname}_vallabels']
    except:
        return  [], []

    num_folders = len(folder_save)

    whole_preds = []
    for ii in range(num_folders):
        valpreds = folder_save[ii][f'{testname}_valpreds']
        whole_preds.append(valpreds)
    whole_preds = np.array(whole_preds)

    avg_preds = np.mean(whole_preds, axis=0)
    return labels, avg_preds

