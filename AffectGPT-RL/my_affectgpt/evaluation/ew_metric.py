'''
This is use for performance evaluation using EW-based metrics.
The main code is draw from .//desktop//main_affectgpt.py
'''

import config
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import glob
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from toolkit.utils.read_files import *
from toolkit.utils.qwen import *
from toolkit.utils.functions import *

from my_affectgpt.evaluation.wheel import calculate_openset_overlap_rate, func_get_name2reason, read_format2raws, \
                                         read_candidate_synonym_merge, calculate_openset_similarity, \
                                         calculate_openset_onehot_hitrate, func_hit_or_not


## get set2names mapping
def read_OVMERDPlus_set2names(split_info_path):

    set2names = {}

    train_names = np.load(split_info_path, allow_pickle=True)['train_names'].tolist()
    test_names  = np.load(split_info_path, allow_pickle=True)['test_names'].tolist()
    whole_names = train_names + test_names
    print (f'{len(train_names)} | {len(test_names)} | {len(whole_names)}')

    set2names = {
        'train': train_names,
        'test':  test_names,
        'whole': whole_names,
    }
    return set2names


# load the model once to avoid repeated calls later
def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


## clue merge batch calling
def clue_merge_batchcalling(name2reason=None, store_npz=None, modelname=None, name2subtitle=None,
                            llm=None, tokenizer=None, sampling_params=None):
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        ## process by batch
        batch_responses = [""] * len(batch_names)
        batch_reasons   = [name2reason[name]   for name in batch_names]
        batch_subtitles = [name2subtitle[name] for name in batch_names]
        ## only process the part where subtitle != ''
        process_indexes, batch_reasons_new, batch_subtitles_new = [], [], []
        for ii, (subtitle, reason) in enumerate(zip(batch_subtitles, batch_reasons)):
            if subtitle != '':
                process_indexes.append(ii)
                batch_reasons_new.append(reason)
                batch_subtitles_new.append(subtitle)
            elif subtitle == '' and reason == '':
                batch_responses[ii] = "Unable to judge the emotional state."
            elif subtitle == '' and reason != '':
                batch_responses[ii] = reason
        ## call
        batch_responses_new = reason_merge_qwen(batch_reasons=batch_reasons_new, batch_subtitles=batch_subtitles_new,
                                                llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
        ## merge the results
        for (index, response) in zip(process_indexes, batch_responses_new):
            batch_responses[index] = response
        print (batch_responses)

        ## merge after finishing the results of one batch
        whole_responses.extend(batch_responses)
    
    ## storage
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)


def subtitle_reasoning_batchcalling(name2subtitle=None,
                                    store_npz=None,
                                    modelname=None, llm=None, tokenizer=None, sampling_params=None):
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## main process
    whole_names, whole_responses = list(name2subtitle.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        ## process by batch
        batch_responses = [""] * len(batch_names)
        batch_reasons   = [""] * len(batch_names)
        batch_subtitles = [name2subtitle[name] for name in batch_names]
        ## only process the part where subtitle != ''
        process_indexes, batch_reasons_new, batch_subtitles_new = [], [], []
        for ii, (subtitle, reason) in enumerate(zip(batch_subtitles, batch_reasons)):
            if subtitle != '':
                process_indexes.append(ii)
                batch_reasons_new.append(reason)
                batch_subtitles_new.append(subtitle)
            else:
                batch_responses[ii] = "Unable to judge the emotional state from the subtitle."
        ## call
        batch_responses_new = reason_merge_qwen(batch_reasons=batch_reasons_new, batch_subtitles=batch_subtitles_new,
                                                llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
        ## merge the results
        for (index, response) in zip(process_indexes, batch_responses_new):
            batch_responses[index] = response
        print (batch_responses)

        ## merge after finishing the results of one batch
        whole_responses.extend(batch_responses)
    
    ## storage
    if store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses
    

## reason -> ov labels
# inputtype: for formatted output, whether the metric is computed from the <think> part or from the <answer> part; for AffectGPT the two are the same, but for R1-Omni they are different
def extract_openset_batchcalling(reason_root=None, reason_npz=None, update_npz=None, reason_csv=None, name2reason=None,
                                 store_root=None, store_npz=None, 
                                 modelname=None, llm=None, tokenizer=None, sampling_params=None,
                                 inputtype='answer'):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if reason_root is not None:
        name2reason = func_get_name2reason(reason_root)
    elif reason_npz is not None:
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()
    elif reason_csv is not None:
        names = func_read_key_from_csv(reason_csv, 'name')
        reasons = func_read_key_from_csv(reason_csv, 'reason')
        name2reason = {}
        for (name, item) in zip(names, reasons):
            name2reason[name] = item
    elif update_npz is not None:
        name2reason = {}
        filenames = np.load(update_npz, allow_pickle=True)['filenames']
        fileitems = np.load(update_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2reason[name] = item
    
    ## => temporarily added processing of name2reason, keeping only the middle answer part for verification
    ## => if matched, use answer; if not matched, use the original
    for name in name2reason:
        reason = name2reason[name]
        if inputtype == 'answer': # default
            match = re.search(r'<answer>(.*?)</answer>', reason)
            if match: reason = match.group(1)
        elif inputtype == 'think':
            match = re.search(r'<think>(.*?)</think>', reason)
            if match: reason = match.group(1)
        name2reason[name] = reason

    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_openset_qwen(llm=llm, tokenizer=tokenizer,
                                                 sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    if store_root is not None:
        if not os.path.exists(store_root):
            os.makedirs(store_root)
        # save to folder
        for (name, response) in zip(whole_names, whole_responses):
            save_path = os.path.join(store_root, f'{name}.npy')
            np.save(save_path, response)
    elif store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses


## reason -> ov labels [sample-wise calling]
def extract_openset_samplecalling(reasons=None, 
                                  modelname=None, model=None, tokenizer=None):
    
    ## load model
    if model is None and tokenizer is None:
        model_path = config.PATH_TO_LLM[modelname]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    ## main process
    responses = []
    for reason in reasons:
        response = reason_to_openset_qwen(model=model, tokenizer=tokenizer,
                                          reason=reason)
        responses.append(response)
    return responses
    

def translate_chi2eng_batchcalling(input_path=None, save_path=None, 
                                   modelname=None, llm=None, tokenizer=None, sampling_params=None):

    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    name2reason = np.load(input_path, allow_pickle=True)['name2reason'].tolist()
    
    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = translate_chi2eng_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    name2reason_update = {}
    for (name, response) in zip(whole_names, whole_responses):
        name2reason_update[name] = response
    np.savez_compressed(save_path, name2reason=name2reason_update)


## this part is for batch processing
def translate_eng2chi_batchcalling(input_path=None, name2reason=None,
                                   save_path=None, 
                                   modelname=None, llm=None, tokenizer=None, sampling_params=None):

    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if name2reason is None:
        name2reason = np.load(input_path, allow_pickle=True)['name2reason'].tolist()
    
    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = translate_eng2chi_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    name2reason_update = {}
    for (name, response) in zip(whole_names, whole_responses):
        name2reason_update[name] = response
    if save_path is not None:
        np.savez_compressed(save_path, name2reason=name2reason_update)
    else:
        return name2reason_update
    

## ov labels -> sentiment
def openset_to_sentiment_batchcalling(openset_npz=None, name2openset=None, 
                                      store_npz=None,
                                      modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if openset_npz is not None:
        name2openset = {}
        filenames = np.load(openset_npz, allow_pickle=True)['filenames']
        fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2openset[name] = item

    ## main process
    whole_names, whole_responses = list(name2openset.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2openset[name] for name in batch_names]
        batch_responses = openset_to_sentiment_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                    batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage [allowed to not store results and return them directly]
    if store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses



def reason_to_valence_batchcalling(label_csv=None, name2reason=None, 
                                   store_npz=None,
                                   modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    if label_csv is not None:
        name2reason = {}
        names = func_read_key_from_csv(label_csv, 'name')
        reasons = func_read_key_from_csv(label_csv, 'reason')
        for (name, reason) in zip(names, reasons):
            name2reason[name] = reason

    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_valence_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage [allowed to not store results and return them directly]
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)


## reason -> onehot labels
def onehot_extraction_replacement_batchcalling(openset_npz=None, store_npz=None, candidates=None, 
                                           modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2openset
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    name2openset = {}
    for (name, item) in zip(filenames, fileitems):
        name2openset[name] = item

    ## main process
    whole_names, whole_responses = list(name2openset.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2openset[name] for name in batch_names]
        ## this only applies to the mer2023 dataset
        batch_responses = openset_to_onehot_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)
        

# combine the above two steps into one: reason -> onehot labels
def clue_to_onehot_batchcalling(reason_npz=None, update_npz=None, store_npz=None, modelname=None, 
                                llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if reason_npz is not None:
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()
    elif update_npz is not None:
        name2reason = {}
        filenames = np.load(update_npz, allow_pickle=True)['filenames']
        fileitems = np.load(update_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2reason[name] = item

    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_onehot_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                batch_reasons=batch_reasons)
        print (batch_responses)
        whole_responses.extend(batch_responses)
    
    ## storage
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)


# process using the rank operation
def rank_batchcalling(reason_npz=None, update_npz=None, store_npz=None, 
                      modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if reason_npz is not None:
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()
    elif update_npz is not None:
        name2reason = {}
        filenames = np.load(update_npz, allow_pickle=True)['filenames']
        fileitems = np.load(update_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2reason[name] = item

    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_rank_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                batch_reasons=batch_reasons)
        print (batch_responses)
        whole_responses.extend(batch_responses)
    
    ## storage
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)


# map all chinese emotion labels -> english emotion labels
def get_chi2eng_label_mapping(emotions, modelname=None):
    # 1. load model
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # 2. batch-wise conversion
    whole_responses = []
    batches_emotions = split_list_into_batch(emotions, batchsize=2)
    for batch_emotions in batches_emotions:
        batch_responses = translate_chi2eng_qwen(llm=llm, tokenizer=tokenizer,
                                                 sampling_params=sampling_params, 
                                                 batch_reasons=batch_emotions)
        whole_responses.extend(batch_responses)
    
    chi2eng_mapping = {}
    for chi, eng in zip(emotions, whole_responses):
        chi2eng_mapping[chi] = eng
    
    return chi2eng_mapping
    

# function: input [gt, openset]; output: the average result under 12 EW-based metrics
def wheel_metric_calculation(gt_root=None, gt_csv=None, name2gt=None, 
                             openset_root=None, openset_npz=None, name2pred=None, 
                             process_names=None, inter_print=True, level='level1'):

    if level == 'level1':
        candidate_metrics = [
                            'case3_wheel1_level1',
                            'case3_wheel2_level1',
                            'case3_wheel3_level1',
                            'case3_wheel4_level1',
                            'case3_wheel5_level1',
                            ]
    elif level == 'level2':
        candidate_metrics = [
                            'case3_wheel1_level2',
                            'case3_wheel2_level2',
                            'case3_wheel3_level2',
                            'case3_wheel4_level2',
                            'case3_wheel5_level2',
                            ]

    # compute this value for each metric
    whole_scores = []
    for metric in candidate_metrics:
        precision, recall = calculate_openset_overlap_rate(gt_root=gt_root,
                                                           gt_csv=gt_csv,
                                                           name2gt=name2gt,
                                                           openset_root=openset_root, 
                                                           openset_npz=openset_npz, 
                                                           name2pred=name2pred,
                                                           process_names=process_names, 
                                                           metric=metric,
                                                           inter_print=inter_print)
        # this case occurs with very low probability
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)
        whole_scores.append([fscore, precision, recall])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores


def hitrate_metric_calculation(name2gt=None, openset_root=None, openset_npz=None, name2pred=None, inter_print=True):

    # use M-avg as the main metric [all mapped to level1 labels]
    candidate_metrics = [
                        'case3_wheel1_level1',
                        'case3_wheel2_level1',
                        'case3_wheel3_level1',
                        'case3_wheel4_level1',
                        'case3_wheel5_level1',
                        ]

    # compute this value for each metric
    whole_scores = []
    for metric in candidate_metrics:
        hitrate, mscore = calculate_openset_onehot_hitrate(name2gt=name2gt,
                                                           openset_root=openset_root,
                                                           openset_npz=openset_npz,
                                                           name2pred=name2pred,
                                                           metric=metric,
                                                           inter_print=inter_print)
        whole_scores.append([hitrate, mscore])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores



def func_process_for_error_cases(pred_list, candiate_labels):
    for item in pred_list:
        if item in candiate_labels:
            return item
    return pred_list[0]


## compute traditional classification metrics
def cls_metric_calculation(name2gt, openset_npz):

    # read name2pred
    names = np.load(openset_npz, allow_pickle=True)['filenames']
    items = np.load(openset_npz, allow_pickle=True)['fileitems']
    name2pred = {}
    for (name, item) in zip(names, items):
        name2pred[name] = item

    # get candidate labels and emo2id_mapping
    candiate_labels = sorted(set([name2gt[name] for name in name2gt]))
    print (f'candidate labels: {candiate_labels}')
    emo2id_mapping = {}
    for ii, emo in enumerate(candiate_labels):
        emo2id_mapping[emo] = ii

    # calculate (accuracy, recall) two values
    gts, preds = [], []
    for name in name2gt:
        gt = name2gt[name]
        
        ## post-process the pred results
        pred = string_to_list(name2pred[name])
        if len(pred) == 1:
            pred = pred[0]
        elif len(pred) == 0:
            pred = 'neutral'
        else: # this case is rare, the list contains multiple elements
            pred = func_process_for_error_cases(pred, candiate_labels)
        
        ## pred metric computation
        if pred not in candiate_labels:
            print (pred)
            continue
        else:
            gts.append(emo2id_mapping[gt])
            preds.append(emo2id_mapping[pred])
    
    print ('process number (before filter): ', len(name2pred))
    print ('process number (after filter):  ', len(gts))
    fscore = f1_score(gts, preds, average='weighted')
    return  fscore


## temporary test: extract the ov labels of emer coarse
def ov_extraction_batchcalling(name2reason, store_npz):
    
    ## load model
    model_path = config.PATH_TO_LLM['Qwen2']
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_openset_qwen(llm=llm, tokenizer=tokenizer,
                                                 sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    np.savez_compressed(store_npz,
                        filenames=whole_names,
                        fileitems=whole_responses)


# ICLR additional experiment: compute the similarity score between set1 and set2


## new code computes the similarity of the two
def similarity_score_calculation(name2pred1, name2pred2, level='level1'):

    if level == 'level1':
        candidate_metrics = [
                            'case3_wheel1_level1',
                            'case3_wheel2_level1',
                            'case3_wheel3_level1',
                            'case3_wheel4_level1',
                            'case3_wheel5_level1',
                            ]
    elif level == 'level2':
        candidate_metrics = [
                            'case3_wheel1_level2',
                            'case3_wheel2_level2',
                            'case3_wheel3_level2',
                            'case3_wheel4_level2',
                            'case3_wheel5_level2',
                            ]
    
    # compute the result under each metric
    whole_scores = []
    for metric in candidate_metrics:
        similarity = calculate_openset_similarity(name2pred1=name2pred1,
                                                  name2pred2=name2pred2,
                                                  metric=metric)
        whole_scores.append(similarity)
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores