'''
This is the main process for emotion wheel. 
1. read emotion wheel.csv
2. Get none, adj, and other manner labels; 
3. calculate overlap rate with existing 301 OV labels; 
4. calculate set-level accuracy and recall based on emotion wheels.
'''
import os
import glob
import scipy
import numpy as np
import pandas as pd

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *

main_root = './emotion_wheel'

## for clash for windows
# set http_proxy=http://127.0.0.1:7890
# set https_proxy=http://127.0.0.1:7890


#############################################
###### read emotion words from all emotion wheels ######
#############################################
# read xlsx and convert it into map format
def read_wheel_to_map(xlsx_path):
    store_map = {}
    level1, level2, level3 = "", "", ""

    df = pd.read_excel(xlsx_path)
    for _, row in df.iterrows():
        row_level1 = row['level1']
        row_level2 = row['level2']
        row_level3 = row['level3']

        # update level1, level2, level3
        if not pd.isna(row_level1):
           level1 = row_level1
        if not pd.isna(row_level2):
           level2 = row_level2
        if not pd.isna(row_level3):
           level3 = row_level3
        
        # store into store_map [needs preprocessing before storing]
        level1 = level1.lower().strip() 
        level2 = level2.lower().strip()
        level3 = level3.lower().strip()
        if level1 not in store_map:
           store_map[level1] = {}
        if level2 not in store_map[level1]:
           store_map[level1][level2] = []
        store_map[level1][level2].append(level3)
    return store_map

# merging all emotion wheels can generate 248 candidate words => 247 candidate labels, related to my correction of fearful
def convert_all_wheels_to_candidate_labels():
    candidate_labels = []
    for xlsx_path in glob.glob(main_root + '/wheel*.xlsx'):
        store_map = read_wheel_to_map(xlsx_path)
        for level1 in store_map:
           for level2 in store_map[level1]:
              level3 = store_map[level1][level2]
              candidate_labels.append(level1)
              candidate_labels.append(level2)
              candidate_labels.extend(level3)
    candidate_labels = list(set(candidate_labels))
    return candidate_labels
'''
Totally, we can generate 253 emotion-wheel labels
'''


###########################################
###### augment labels on the emotion wheel ######
###########################################
# func: merge two maps, merging the lists of elements in the maps
def func_merge_map(map1, map2):
    all_items = list(map1.keys()) + list(map2.keys())

    merge_map = {}
    for item in all_items:
        if item in map1 and item in map2:
            value = list(set(map1[item] + map2[item]))
        elif item not in map1 and item in map2:
            value = map2[item]
        elif item in map1 and item not in map2:
            value = map1[item]
        merge_map[item] = value
    return merge_map


# 1. read the synonym forms of the candidate words
# label2wheel: maps all labels -> labels in the emotion wheel
def read_candidate_synonym_onerun(runname='run1'):

    ## read candidate labels
    wheel_labels = convert_all_wheels_to_candidate_labels()

    ## gain mapping
    label2wheel = {}
    synonym_path = os.path.join(main_root, 'synonym.xlsx')
    df = pd.read_excel(synonym_path)
    for _, row in df.iterrows():

        # build self-mapping
        raw = row[f'word_{runname}'].strip().lower()
        assert raw in wheel_labels, f'error in {raw}' # check openai returns
        if raw not in label2wheel:
            label2wheel[raw] = []
        label2wheel[raw].append(raw)

        # build synonyms -> raw mapping
        synonyms = row[f'synonym_{runname}']
        synonyms = string_to_list(synonyms)
        for synonym in synonyms:
            synonym = synonym.strip().lower()
            if synonym not in label2wheel: 
                label2wheel[synonym] = []
            label2wheel[synonym].append(raw)
    return label2wheel


# 2. call gpt-4o / gpt-3.5 8 times to generate synonyms [this way the emotion word coverage is as complete as possible]
# mapping_merge: maps all labels -> labels in the emotion wheel
def read_candidate_synonym_merge():
    mapping_run1 = read_candidate_synonym_onerun('run1')
    mapping_run2 = read_candidate_synonym_onerun('run2')
    mapping_run3 = read_candidate_synonym_onerun('run3')
    mapping_run4 = read_candidate_synonym_onerun('run4')
    mapping_run5 = read_candidate_synonym_onerun('run5') 
    mapping_run6 = read_candidate_synonym_onerun('run6') 
    mapping_run7 = read_candidate_synonym_onerun('run7') 
    mapping_run8 = read_candidate_synonym_onerun('run8') 
    mapping_merge = func_merge_map(mapping_run1,  mapping_run2)
    mapping_merge = func_merge_map(mapping_merge, mapping_run3)
    mapping_merge = func_merge_map(mapping_merge, mapping_run4)
    mapping_merge = func_merge_map(mapping_merge, mapping_run5)
    mapping_merge = func_merge_map(mapping_merge, mapping_run6)
    mapping_merge = func_merge_map(mapping_merge, mapping_run7)
    mapping_merge = func_merge_map(mapping_merge, mapping_run8)
    print (f'label number: {len(mapping_merge)}') 
    return mapping_merge
# expand the 253 emotion-wheel words to 1255 words with their synonyms

## build the mapping from all forms -> new_labels
# 3. call the API to generate different forms of all words
def generate_different_format_for_words():

    save_root_format1 = os.path.join(main_root, 'format1')
    save_root_format2 = os.path.join(main_root, 'format2')
    if not os.path.exists(save_root_format1): os.makedirs(save_root_format1)
    if not os.path.exists(save_root_format2): os.makedirs(save_root_format2)

    # check the generated new labels
    raw_labels = convert_all_wheels_to_candidate_labels()    # 253
    new_labels = list(read_candidate_synonym_merge().keys()) # 1255
    print (f'level1 number: {len(raw_labels)}')
    print (f'level2 number: {len(new_labels)}')
    for label in raw_labels: assert label in new_labels

    # generate different forms of the labels
    for label in new_labels:
        # for case1
        save_path = os.path.join(save_root_format1, f'{label}.npy')
        if os.path.exists(save_path): continue
        outputs = get_different_format(label, prompt_type='case1')
        np.save(save_path, outputs)

        # for case2
        save_path = os.path.join(save_root_format2, f'{label}.npy')
        if os.path.exists(save_path): continue
        outputs = get_different_format(label, prompt_type='case2')
        np.save(save_path, outputs)

# 4. build the mapping from all forms -> augmented labels
def get_raw2format(format_root):
    raw2formats = {}
    for file_path in glob.glob(format_root + '/*'):
        raw = os.path.basename(file_path)[:-4].lower().strip()
        formats = np.load(file_path, allow_pickle=True)
        formats = string_to_list(formats)
        formats = [label.lower().strip() for label in formats]

        # means label and raw have different word counts, so this needs attention
        formats_new = []
        for format in formats:
            if len(format.split())    != len(raw.split()):    continue
            if len(format.split('-')) != len(raw.split('-')): continue
            formats_new.append(format)
        
        raw2formats[raw] = formats_new
    return raw2formats

# 5. convert format into a csv file, convenient for finding errors
def merge_format_to_csv():
    raw2formats1 = get_raw2format('emotion_wheel/format1')
    raw2formats2 = get_raw2format('emotion_wheel/format2')
    raw2formats  = func_merge_map(raw2formats1, raw2formats2)
    print (f'augmented labels: {len(raw2formats)}')

    # remove duplicated
    for raw in raw2formats:
        formats = raw2formats[raw]
        formats = list(set(formats))
        formats = ",".join(formats)
        raw2formats[raw] = formats
    
    # convert to csv
    raws = [raw for raw in raw2formats]
    csv_path = 'emotion_wheel/format.csv'
    func_write_key_to_csv(csv_path, raws, raw2formats, ['format'])


def read_format2raws():
    format2raws = {}

    format_path = 'emotion_wheel/format.csv'
    raws = func_read_key_from_csv(format_path, 'name')
    formats = func_read_key_from_csv(format_path, 'format')
    for raw, format in zip(raws, formats):

        # 1. build the mapping between format and raw
        format = string_to_list(format)
        for format_item in format:
            if format_item not in format2raws:
                format2raws[format_item] = []
            format2raws[format_item].append(raw)
        
        # 2. build the mapping between raw and raw
        if raw not in format2raws:
            format2raws[raw] = []
        format2raws[raw].append(raw)
    print (len(format2raws))
    return format2raws
# expand the 1255 words to 7386 words with the format augmentation
# => the two steps above turn 255 words into 7386 words, a nearly 40x increase

#############################################################################
######## test the probability that all candidate labels in EMER-Fine appear in the augmented vocabulary ########
#############################################################################

# 1. read all candiate labels in emer-fine
def read_candiate_labels_from_emer_fine():
    whole_candidates = []

    label_csv = 'manual_results_step10/check-result-openset-chi-translate.csv'
    opensets = func_read_key_from_csv(label_csv, 'openset')
    for openset in opensets:
        openset = string_to_list(openset)
        whole_candidates.extend(openset)
    
    # remove space and strip
    whole_candidates = [label.lower().strip() for label in whole_candidates]
    whole_candidates = list(set(whole_candidates))
    print (f'emer fine candiate number: {len(whole_candidates)}') # 248
    return whole_candidates


# 2. read all candidate labels for MLLMs
def read_all_ov_labels_from_LLMs():
    main_root = './'
    
    output_roots = [
        (os.path.join(main_root, 'Ground-truth/update-reason-multi-chi'),   'Multi-GT & $\\surd$  & $\\surd$  & $\\surd$'),
        (os.path.join(main_root, 'Otter-main/output-reason-7b-nosubtitle-chi'), 'Otter(v)  \\cite{li2023otter}   & $\\times$  & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'Video-ChatGPT-main/output-reason-7b-nosubtitle-chi'), 'Video-ChatGPT(v) \\cite{maaz2023video} & $\\times$ & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'Video-LLaMA-main/output-reason-7b-nosubtitle-chi'),   'Video-LLaMA(v) \\cite{zhang2023video}  & $\\times$ & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'Ask-Anything-main/video_chat/output-reason-7b-nosubtitle-chi'),   'VideoChat(v)  \\cite{li2023videochat} & $\\times$  & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'Ask-Anything-main/video_chat2/output-reason-7b-nosubtitle-chi'),  'VideoChat2(v) \\cite{li2024mvbench}   & $\\times$  & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'mPLUG-Owl/output-reason-7b-nosubtitle-chi'),              'mPLUG-Owl(v) \\cite{ye2023mplug}   & $\\times$  & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'SALMONN/output-reason-7b-nosubtitle-chi'),    'SALMONN(a) \\cite{tang2023salmonn}      & $\\times$  & $\\times$ & $\\surd$'),
        (os.path.join(main_root, 'Qwen-main/output-reason-7b-nosubtitle-chi'),  'Qwen-Audio(a) \\cite{chu2023qwen}   & $\\times$  & $\\times$ & $\\surd$'),
        (os.path.join(main_root, 'Video-LLaVA-main/output-reason-7b-nosubtitle-chi'),              'Video-LLaVA(v) \\cite{lin2023video}   & $\\times$  & $\\surd$ & $\\times$'),
        (os.path.join(main_root, 'LLaMA-VID-main/output-reason-7b-nosubtitle-chi'),            'LLaMA-VID(v) \\cite{li2023llama}    & $\\times$  & $\\surd$  & $\\times$'),
        (os.path.join(main_root, 'Chat-UniVi-main/output-reason-7b-nosubtitle-chi'),           'Chat-UniVi(v) \\cite{jin2023chat}   & $\\times$  & $\\surd$  & $\\times$'),
        (os.path.join(main_root, 'OneLLM-main/output-reason-7b-a-chi'),                       'OneLLM(a) \\cite{han2023onellm}    & $\\times$   & $\\times$ & $\\surd$'),
        (os.path.join(main_root, 'OneLLM-main/output-reason-7b-v-chi'),                       'OneLLM(v) \\cite{han2023onellm}    & $\\times$   & $\\surd$  & $\\times$'),
        (os.path.join(main_root, 'SECap-main/output-reason-7b-nosubtitle-chi'),   'SECap & $\\surd$   & $\\times$ & $\\surd$'),
    ]

    whole_candidates = []
    for chi_root, _ in output_roots:
        assert chi_root.endswith('chi')

        ## MLLMs: all use the results under the S2 subtitle fusion strategy
        if chi_root.find('nosubtitle')!=-1 or chi_root.find('-a-')!=-1 or chi_root.find('-v-')!=-1 or chi_root.find('-av-')!=-1:
            for run_time in [0, 1]:
                reason_root = chi_root[:-4] + '-addsub-eng' 
                openset_root = reason_root + f'-run{run_time}-openset'
                print (openset_root)
                for openset_path in glob.glob(openset_root + '/*'):
                    labels = string_to_list(np.load(openset_path).tolist())
                    whole_candidates.extend(labels)

        ## some models read the original results
        else:
            for run_time in [0, 1]:
                reason_root = chi_root[:-4] + '-eng'
                openset_root = reason_root + f'-run{run_time}-openset'
                print (openset_root)
                for openset_path in glob.glob(openset_root + '/*'):
                    labels = string_to_list(np.load(openset_path).tolist())
                    whole_candidates.extend(labels)
    
    whole_candidates = [label.lower().strip() for label in whole_candidates]
    whole_candidates = list(set(whole_candidates))
    print (f'label number: {len(whole_candidates)}') # 1003 labels
    return whole_candidates


## count how many labels in emer-fine appear in the candidate labels
def check_emer_fine_overlap_with_wheel():
    
    format2raws = read_format2raws()
    print (f'vocabulary size: {len(format2raws)}') # 7386

    whole_emerfine = read_candiate_labels_from_emer_fine()
    print (f'emer fine labels: {len(whole_emerfine)}') # 248
    exist_num = 0
    for label in whole_emerfine:
        if label in format2raws:
            exist_num += 1
    print (exist_num, len(whole_emerfine)) # 178/248 => about 70%

    whole_mllms = read_all_ov_labels_from_LLMs()
    print (f'mllms labels: {len(whole_mllms)}')
    exist_num = 0
    for label in whole_mllms:
        if label in format2raws:
            exist_num += 1
    print (exist_num, len(whole_mllms)) # 443/877 => about 50%
# vocabulary size: 7386
# about 70% of emer_fine is in the vocabulary
# about 50% of mllm is in the vocabulary
# so the mllm labels are dirtier and contain many non-emotion words, which matches what we saw before


#########################################################################
######## use the obtained mapping to measure the overlap between gt and pred openset ########
## => all metrics directly remove elements not in the vocabulary [unified processing, convenient for later comparison]
##########################################################################
def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


# case1: the computation depends only on the result under format_mapping
def func_backward_case1(label, format_mapping, raw_mapping=None, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    return stage1_unique


# case2: the core is to guarantee uniqueness during the backward process
def func_backward_case2(label, format_mapping, raw_mapping, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    stage2_labels = raw_mapping[stage1_unique]
    assert isinstance(stage2_labels, list)
    stage2_unique = sorted(stage2_labels)[0]
    return stage2_unique


def func_get_wheel_cluster(wheel='wheel1', level='level1'):
    print (f'process on wheel: {wheel} level: {level}')
    xlsx_path = f'emotion_wheel/{wheel}.xlsx'
    emotion_wheel = read_wheel_to_map(xlsx_path)
    print (emotion_wheel.keys())
    wheel_map = {}

    # 1. cluster everything to level1
    if level == 'level1':
        for level1 in emotion_wheel:
            wheel_map[level1] = level1
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level1
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level1

    # 2. cluster everything to level2
    elif level == 'level2':
        for level1 in emotion_wheel:
            wheel_map[level1] = sorted(emotion_wheel[level1])[0] # level1 maps to a fixed level2
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level2
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level2
    return wheel_map

## function 3: introduce the emotion wheel for evaluation
def func_backward_case3(label, format_mapping, raw_mapping, wheel_map):
    if label not in format_mapping:
        return ""
    
    level1_whole = []
    for format in format_mapping[label]:
        for raw in raw_mapping[format]:
            level1_whole.append(raw)
    
    for level1 in sorted(level1_whole): # guarantees uniqueness of the result
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""


# metric: ['case1', 'case2', 'case3'] presents a hierarchical clustering result, showing the necessity of each level
def func_map_label_to_synonym(mlist, format_mapping, raw_mapping, wheel_map, metric='case1'):
    new_mlist = []
    for label in mlist:
        if metric.startswith('case1'): label = func_backward_case1(label, format_mapping)
        if metric.startswith('case2'): label = func_backward_case2(label, format_mapping, raw_mapping)
        if metric.startswith('case3'): label = func_backward_case3(label, format_mapping, raw_mapping, wheel_map)
        if label == '': continue # if no backward word is found, remove it
        new_mlist.append(label)
    return new_mlist

# compute the overlap rate of one (gt_root, openset_root) under metric
# => code modified: support input files of different formats, and support computing the result for only a subset of samples
# => code modified: support reading format_mapping and raw_mapping from outside
def calculate_openset_overlap_rate(gt_root, openset_root=None, metric='case1', openset_npz=None, process_names=None, 
                                   format_mapping=None, raw_mapping=None, gt_csv=None):

    # read gt info
    if gt_root is not None:
        name2gt = func_get_name2reason(gt_root)
    elif gt_csv is not None:
        name2gt = {}
        names = func_read_key_from_csv(gt_csv, 'name')
        gts   = func_read_key_from_csv(gt_csv, 'openset')
        for (name, gt) in zip(names, gts):
            name2gt[name] = gt

    # read openset => (name2pred)
    if openset_root is not None:
        name2pred = func_get_name2reason(openset_root)
    elif openset_npz is not None:
        names = np.load(openset_npz)['filenames']
        items = np.load(openset_npz)['fileitems']
        name2pred = {}
        for (name, item) in zip(names, items):
            name2pred[name] = item

    # process_names => (whole) / (subset)
    if process_names is None:
        process_names = [name for name in name2gt]

    # read all mapping 
    if format_mapping is None:
        format_mapping = read_format2raws()          # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge() # level2 -> level1
    if metric.startswith('case3'): # level1 -> cluster center
        _, wheelname, levelname = metric.split('_')
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # calculate (accuracy, recall) two values
    accuracy, recall = [], []
    for name in process_names:      
       
        # remove synonyms in gt and pred
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))

        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))
       
        if len(gt) == 0: continue # skip such samples directly
        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred)/len(pred))
            recall.append(len(gt & pred)/len(gt))
    print ('process number (after filter): ', len(accuracy))
    avg_accuracy, avg_recall = np.mean(accuracy), np.mean(recall)
    print (f'avg acc: {avg_accuracy} avg recall: {avg_recall}')
    return  avg_accuracy, avg_recall


gt_root = 'manual_results_step10/check-result-openset-chi-translate'
output_roots = [
    ('Ground-truth/update-reason-multi-chi',                           'Multi-GT & $\\surd$  & $\\surd$  & $\\surd$'),
    ('Otter-main/output-reason-7b-nosubtitle-chi',                     'Otter(v)  \\cite{li2023otter}   & $\\times$  & $\\surd$ & $\\times$'),
    ('Video-ChatGPT-main/output-reason-7b-nosubtitle-chi',             'Video-ChatGPT(v) \\cite{maaz2023video} & $\\times$ & $\\surd$ & $\\times$'),
    ('Video-LLaMA-main/output-reason-7b-nosubtitle-chi',               'Video-LLaMA(v) \\cite{zhang2023video}  & $\\times$ & $\\surd$ & $\\times$'),
    ('Ask-Anything-main/video_chat/output-reason-7b-nosubtitle-chi',   'VideoChat(v)  \\cite{li2023videochat} & $\\times$  & $\\surd$ & $\\times$'),
    ('Ask-Anything-main/video_chat2/output-reason-7b-nosubtitle-chi',  'VideoChat2(v) \\cite{li2024mvbench}   & $\\times$  & $\\surd$ & $\\times$'),
    ('mPLUG-Owl/output-reason-7b-nosubtitle-chi',                      'mPLUG-Owl(v) \\cite{ye2023mplug}   & $\\times$  & $\\surd$ & $\\times$'),
    ('SALMONN/output-reason-7b-nosubtitle-chi',                        'SALMONN(a) \\cite{tang2023salmonn}      & $\\times$  & $\\times$ & $\\surd$'),
    ('Qwen-main/output-reason-7b-nosubtitle-chi',                      'Qwen-Audio(a) \\cite{chu2023qwen}   & $\\times$  & $\\times$ & $\\surd$'),
    ('Video-LLaVA-main/output-reason-7b-nosubtitle-chi',               'Video-LLaVA(v) \\cite{lin2023video}   & $\\times$  & $\\surd$ & $\\times$'),
    ('LLaMA-VID-main/output-reason-7b-nosubtitle-chi',                 'LLaMA-VID(v) \\cite{li2023llama}    & $\\times$  & $\\surd$  & $\\times$'),
    ('Chat-UniVi-main/output-reason-7b-nosubtitle-chi',                'Chat-UniVi(v) \\cite{jin2023chat}   & $\\times$  & $\\surd$  & $\\times$'),
    ('OneLLM-main/output-reason-7b-a-chi',                             'OneLLM(a) \\cite{han2023onellm}    & $\\times$   & $\\times$ & $\\surd$'),
    ('OneLLM-main/output-reason-7b-v-chi',                             'OneLLM(v) \\cite{han2023onellm}    & $\\times$   & $\\surd$  & $\\times$'),
    ('SECap-main/output-reason-7b-nosubtitle-chi',                     'SECap & $\\surd$   & $\\times$ & $\\surd$'),
]
def calculate_scores_for_MLLMs(metric):
    whole_store, store_path = {}, f'./whole_store_{metric}.npy'
    for chi_root, _ in output_roots:
        assert chi_root.endswith('chi')

        ## MLLMs: all use the results under the S2 subtitle fusion strategy
        if chi_root.find('nosubtitle')!=-1 or chi_root.find('-a-')!=-1 or chi_root.find('-v-')!=-1 or chi_root.find('-av-')!=-1:
            for run_time in [0, 1]:
                reason_root = chi_root[:-4] + '-addsub-eng' 
                openset_root = reason_root + f'-run{run_time}-openset'
                print (openset_root)
                accuracy, recall = calculate_openset_overlap_rate(gt_root, openset_root, metric)
                if reason_root not in whole_store: whole_store[reason_root] = []
                whole_store[reason_root].append([accuracy, recall])
        
        ## results computed under non-MLLMs
        else:
            for run_time in [0, 1]:
                reason_root = chi_root[:-4] + '-eng'
                openset_root = reason_root + f'-run{run_time}-openset'
                print (openset_root)
                accuracy, recall = calculate_openset_overlap_rate(gt_root, openset_root, metric)
                if reason_root not in whole_store: whole_store[reason_root] = []
                whole_store[reason_root].append([accuracy, recall])
    np.save(store_path, whole_store)


## use this result to analyze the consistency of trend changes, thereby reducing the evaluation cost
candidate_metrics = [
                     'case1', 'case2',
                     'case3_wheel1_level1', 'case3_wheel1_level2',
                     'case3_wheel2_level1', 'case3_wheel2_level2',
                     'case3_wheel3_level1', 'case3_wheel3_level2',
                     'case3_wheel4_level1', 'case3_wheel4_level2',
                     'case3_wheel5_level1', 'case3_wheel5_level2',
                     ]

## temporary test: metric stability analysis [the metrics are very stable, which already rules out the influence of the self-defined metric on performance]

## test and store the results of all MLLMs


#####################################
## summary of the results, computing the correlation between different metrics
#####################################
model_order = [
    ('Qwen-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Otter-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Video-LLaMA-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Ask-Anything-main/video_chat/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('SECap-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Video-LLaVA-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('SALMONN/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Ask-Anything-main/video_chat2/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Video-ChatGPT-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('LLaMA-VID-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('mPLUG-Owl/output-reason-7b-nosubtitle-addsub-eng', '_'),
    ('Chat-UniVi-main/output-reason-7b-nosubtitle-addsub-eng', '_'),
]

def show_performance_of_different_metrics():

    # gain statistics
    whole_results = []
    for metric in ['case3_wheel3_level1', 'case3_wheel3_level2', 'case3_wheel4_level1', 'case3_wheel4_level2', 'case3_wheel5_level1', 'case3_wheel5_level2']:
        metric_results = []
        save_path = f'whole_store_{metric}.npy'
        model2scores = np.load(save_path, allow_pickle=True).tolist()
        for model, _ in model_order:
            # read scores
            scores = model2scores[model]
            # convert scores
            scores = np.array(scores)[:, :2].astype('float')
            scores = scores * 100 # [2, 2]
            
            ## for fscore
            fscore = 2 * (scores[:, 0] * scores[:, 1]) / (scores[:, 0] + scores[:, 1]) # [2, 1]
            fscore = fscore[:, np.newaxis]
            scores = np.concatenate([fscore, scores], axis=1) # [2, 3]

            meanvals = np.mean(scores, axis=0)
            stdvals  = np.std(scores, axis=0)
            final_score, find_std = meanvals[0], stdvals[0]
            print (model, f'%.2f \\pm %.2f' %(final_score, find_std))
            metric_results.append(f'%.2f$\\pm$%.2f' %(final_score, find_std))
        whole_results.append(metric_results)
    
    # show results
    total_xx = len(whole_results)
    total_yy = len(whole_results[0])
    for jj in range(total_yy):
        oneline = []
        for ii in range(total_xx):
            oneline.append(whole_results[ii][jj])
        print (" & ".join(oneline) + '\\\\')


def calculate_metric_similarity():
    
    whole_results1 = f"""Qwen-Audio &38.13$\pm$0.05 & 20.49$\pm$0.01 & 23.37$\pm$0.01 & 43.85$\pm$0.03 & 26.60$\pm$0.01 & 41.52$\pm$0.28 & 26.68$\pm$0.01\\
Otter  &43.51$\pm$0.09 & 22.21$\pm$0.06 & 27.99$\pm$0.02 & 49.75$\pm$0.11 & 33.50$\pm$0.06 & 49.93$\pm$0.11 & 33.04$\pm$0.06\\
Video-LLaMA  &44.73$\pm$0.14 & 23.56$\pm$0.08 & 28.39$\pm$0.17 & 52.90$\pm$0.12 & 36.08$\pm$0.13 & 53.60$\pm$0.04 & 35.33$\pm$0.08\\
VideoChat  &45.53$\pm$0.11& 22.15$\pm$0.00 & 26.24$\pm$0.08 & 47.79$\pm$0.07 & 32.64$\pm$0.07 & 47.76$\pm$0.11 & 32.14$\pm$0.04\\
SECap  &45.72$\pm$0.09 & 26.52$\pm$0.01 & 32.88$\pm$0.03 & 52.26$\pm$0.03 & 37.55$\pm$0.03 & 52.11$\pm$0.03 & 37.71$\pm$0.03\\
Video-LLaVA  &47.07$\pm$0.16 & 25.47$\pm$0.12 & 30.73$\pm$0.11 & 54.65$\pm$0.10 & 37.65$\pm$0.24 & 54.54$\pm$0.02 & 38.25$\pm$0.22\\
SALMONN &47.96$\pm$0.04 & 23.57$\pm$0.02 & 28.83$\pm$0.03 & 54.90$\pm$0.15 & 38.93$\pm$0.15 & 54.29$\pm$0.06 & 37.79$\pm$0.07\\
VideoChat2  &49.07$\pm$0.26 & 26.92$\pm$0.09 & 31.40$\pm$0.10 & 52.38$\pm$0.13 & 36.44$\pm$0.11 & 53.56$\pm$0.13 & 36.91$\pm$0.11\\
Video-ChatGPT  &50.52$\pm$0.06 & 28.99$\pm$0.04 & 34.05$\pm$0.05 & 57.66$\pm$0.04 & 41.48$\pm$0.09 & 57.37$\pm$0.00 & 40.95$\pm$0.08\\
LLaMA-VID  &51.25$\pm$0.09 & 28.28$\pm$0.04 & 32.85$\pm$0.03 & 56.59$\pm$0.04 & 41.22$\pm$0.02 & 57.49$\pm$0.03 & 40.39$\pm$0.04\\
mPLUG-Owl  &52.73$\pm$0.13  & 27.47$\pm$0.17 & 32.47$\pm$0.19 & 57.60$\pm$0.23 & 41.32$\pm$0.04 & 56.32$\pm$0.26 & 40.83$\pm$0.07\\
Chat-UniVi  &53.08$\pm$0.01 & 28.89$\pm$0.02 & 33.23$\pm$0.08 & 57.00$\pm$0.06 & 42.25$\pm$0.04 & 57.50$\pm$0.03 & 42.43$\pm$0.03\\
"""
    
    whole_results2 = f"""Qwen-Audio 		& 39.46$\pm$0.28 & 30.65$\pm$0.01 & 36.64$\pm$0.03 & 27.33$\pm$0.01 & 35.89$\pm$0.08 & 29.66$\pm$0.01 & 31.84
Otter  			& 51.03$\pm$0.04 & 37.12$\pm$0.00 & 47.54$\pm$0.00 & 34.77$\pm$0.00 & 50.51$\pm$0.03 & 35.54$\pm$0.00 & 39.41
Video-LLaMA  	& 47.50$\pm$0.20 & 36.50$\pm$0.25 & 52.97$\pm$0.09 & 35.78$\pm$0.14 & 46.39$\pm$0.12 & 34.77$\pm$0.23 & 40.31
VideoChat  		& 46.78$\pm$0.11 & 34.37$\pm$0.03 & 49.53$\pm$0.15 & 32.82$\pm$0.01 & 45.93$\pm$0.18 & 32.85$\pm$0.04 & 37.58
SECap  			& 50.77$\pm$0.03 & 40.49$\pm$0.03 & 50.43$\pm$0.03 & 38.21$\pm$0.03 & 49.97$\pm$0.03 & 40.25$\pm$0.03 & 42.43 
Video-LLaVA  	& 52.29$\pm$0.05 & 40.58$\pm$0.15 & 52.45$\pm$0.06 & 39.91$\pm$0.13 & 52.97$\pm$0.10 & 39.69$\pm$0.10 & 43.27
SALMONN 		& 56.25$\pm$0.01 & 43.01$\pm$0.02 & 50.53$\pm$0.09 & 38.54$\pm$0.03 & 53.65$\pm$0.04 & 42.09$\pm$0.02 & 43.53
VideoChat2  	& 52.14$\pm$0.23 & 40.57$\pm$0.14 & 50.63$\pm$0.19 & 39.64$\pm$0.18 & 51.37$\pm$0.14 & 39.89$\pm$0.15 & 42.65
Video-ChatGPT  	& 55.50$\pm$0.13 & 44.15$\pm$0.18 & 55.24$\pm$0.02 & 42.42$\pm$0.05 & 52.93$\pm$0.05 & 41.54$\pm$0.14 & 46.02
LLaMA-VID  		& 55.12$\pm$0.05 & 44.06$\pm$0.01 & 56.62$\pm$0.15 & 42.42$\pm$0.03 & 53.03$\pm$0.08 & 41.65$\pm$0.04 & 45.81
mPLUG-Owl  		& 55.67$\pm$0.19 & 43.71$\pm$0.13 & 55.06$\pm$0.17 & 40.67$\pm$0.19 & 54.44$\pm$0.13 & 42.00$\pm$0.18 & 45.63
Chat-UniVi  	& 56.80$\pm$0.01 & 45.66$\pm$0.05 & 55.86$\pm$0.07 & 41.97$\pm$0.09 & 55.81$\pm$0.02 & 43.61$\pm$0.05 & 46.75
"""
    
    # read the data
    whole_scores1 = []
    lines = whole_results1.split('\n')
    lines = [line for line in lines if len(line.strip()) != 0]
    for line in lines:
        line_scores = []
        for item in line.split('&')[1:]:
            line_scores.append(float(item.split('$\pm$')[0]))
        whole_scores1.append(line_scores)

    whole_scores2 = []
    lines = whole_results2.split('\n')
    lines = [line for line in lines if len(line.strip()) != 0]
    for line in lines:
        line_scores = []
        for item in line.split('&')[1:]:
            line_scores.append(float(item.split('$\pm$')[0]))
        whole_scores2.append(line_scores)
    
    # sum all Manual scores
    
    # correlation computation
    whole_scores = np.concatenate([np.array(whole_scores1), np.array(whole_scores2)], axis=1)
    for ii in range(len(whole_scores[0])):
        ii_scores = np.array(whole_scores)[:, ii]
        oo_scores = np.array(whole_scores)[:, 0]
        pcc_score = calculate_pcc(ii_scores, oo_scores)
        print (f'%.3f' %(pcc_score))

# the different metrics correlate strongly with each other, all above 90%


#####################################
## plot the word cloud / long-tail distribution
#####################################
def calculate_onehot_frequency():

    ## read onehot labels
    name2onehot = {}
    label_path = 'manual_results_step10/label-disdim.csv'
    names = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'discrete')
    for (name, label) in zip(names, labels):
        name2onehot[name] = label
    
    ## read processed names
    gt_root = 'manual_results_step10/check-result-openset-chi-translate'
    name2gt = func_get_name2reason(gt_root)
    process_names = list(name2gt.keys())

    ## read wordcount
    wordcount = {}
    for name in process_names:
        onehot = name2onehot[name]
        if onehot not in wordcount:
            wordcount[onehot] = 0
        wordcount[onehot] += 1
    return wordcount


def calculate_gt_ov_frequency():
    wordcount = {}

    gt_root = 'manual_results_step10/check-result-openset-chi-translate'
    name2gt = func_get_name2reason(gt_root)

    for name in name2gt:
        openset = name2gt[name]
        for label in openset:
            if label not in wordcount:
                wordcount[label] = 0
            wordcount[label] += 1

    return wordcount


def calculate_emer_direct_ov_frequency():
    wordcount = {}

    gt_root = 'EMER-Fine-v1\\gt-eng-run1-openset'
    name2gt = func_get_name2reason(gt_root)

    for name in name2gt:
        openset = string_to_list(name2gt[name])
        for label in openset:
            if label not in wordcount:
                wordcount[label] = 0
            wordcount[label] += 1

    return wordcount


# word_frequencies: map between "word -> cound"
def draw_wordcloud(word_frequencies, save_png):
    from wordcloud import WordCloud

    # generate the word cloud using word frequencies
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

    # show the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # turn off the axes
    plt.tight_layout() 
    plt.savefig(save_png, format='png')
    plt.close()

def plot_longtail(word_frequencies, save_png):

    ## sort labels based on frequency
    emotions, frequencies = [], []
    for emotion in word_frequencies:
        emotions.append(emotion)
        frequencies.append(word_frequencies[emotion])
    emotions, frequencies = np.array(emotions), np.array(frequencies)
    indexes = np.argsort(-frequencies)
    frequencies = frequencies[indexes]
    frequencies = frequencies / np.sum(frequencies)
    emotions    = emotions[indexes]

    ## plot in order of frequency high -> low
    xxc = np.arange(0, len(emotions))
    xxc_names = emotions
    yyc = frequencies

    ## main plot
    fontsize = 16
    plt.figure(figsize=(14, 3))
    plt.bar(xxc, yyc, color='orange')
    plt.xlabel('emotion label', fontsize=fontsize)
    plt.ylabel('frequency', fontsize=fontsize)
    plt.xticks([])
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_png, format='png')
    plt.close()
    
#####################################
## plot the label-count distribution
#####################################
# count the number of labels per sample
def calculate_labelnumber_per_sample(openset_root):
    len2count = {}
    name2gt = func_get_name2reason(openset_root)
    for name in name2gt:
        openset = string_to_list(name2gt[name])
        lennum = len(openset)
        if lennum not in len2count:
            len2count[lennum] = 0
        len2count[lennum] += 1
    print (len2count)
    return len2count

def draw_labelnumber_frequency(len2count, color, save_png):
    
    ## gain xx
    alllens = list(len2count.keys())
    minlen = min(alllens) # 1 # min(alllens)
    maxlen = max(alllens) # 9 # max(alllens)
    xx = np.arange(minlen, maxlen + 1)

    ## gain yy
    yy = []
    for item in xx:
        if item in len2count:
            yy.append(len2count[item])
        else:
            yy.append(0)
    yy = np.array(yy)
    yy = yy / np.sum(yy)
    
    ## plot
    fontsize = 16
    plt.figure()
    plt.bar(xx, yy, color=color)
    plt.xlabel('ov label number', fontsize=fontsize)
    plt.ylabel('frequency', fontsize=fontsize)
    plt.xticks(xx, [str(item) for item in xx])
    plt.ylim(0, 0.7)
    plt.xlim(0, 10)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_png, format='png')
    plt.close()

#####################################
## plot the CLUE-M/A/V/T length distribution
# => we found the model has an extra update for CLUE-V, using not the original video-clue but video-clue + extra analysis, so its length distribution does not match my expectation
#####################################
def statistic_clue_lens(clue_root):
    lens = []
    name2clue = func_get_name2reason(clue_root)
    for name in name2clue:
        clue = name2clue[name]
        words = clue.split()
        lens.append(len(words))
    return lens

def draw_clue_lens(clue_root, save_png, binnum=20, color='b'):
    
    data = statistic_clue_lens(clue_root)
    print (f'sample number: {len(data)}')

    plt.figure()

    # => plot histogram
    min_val = min(data)
    max_val = max(data)
    iterval = (max_val - min_val) / binnum
    bins = np.arange(min_val, max_val+iterval, iterval)
    plt.hist(data, bins, density=True, color=color)

    fontsize = 16
    plt.xlabel('Description Length', fontsize=fontsize)
    plt.ylabel('Frequency',          fontsize=fontsize)
    plt.xlim(0, 400)
    plt.ylim(0, 0.04)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_png, format='png')
    plt.close()

