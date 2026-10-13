'''
测试标点符号对于识别结果的影响
'''
import os
import glob
import shutil
import random
import numpy as np

from toolkit.globals import *
from toolkit.utils.read_files import *

import zhon
import string

def is_chinese_character(_char):
    if '\u4e00' <= _char <= '\u9fff':
        return True
    else:
        return False

def is_chinese_punc(_char):
    if _char in zhon.hanzi.punctuation:
        return True
    else:
        return False
    
def is_english_character(_char):
    if 'a' <= _char <= 'z' or 'A' <= _char <= 'Z':
        return True
    else:
        return False

def is_english_punc(_char):
    if _char in string.punctuation:
        return True
    else:
        return False

def is_digit(_char):
    if '0' <= _char <= '9':
        return True
    else:
        return False

def is_space(_char):
    if _char == ' ':
        return True
    else:
        return False

def is_nextline(_char):
    if _char == '\n':
        return True
    else:
        return False

## 找到所有字符信息 => 这几种已经基本覆盖所有字符了
def func_find_all_chars(sentence):
    for _char in sentence:
        if  not is_chinese_character(_char) and not is_chinese_punc(_char) and \
                not is_english_character(_char) and not is_english_punc(_char) and \
                not is_digit(_char) and not is_space(_char) and not is_nextline(_char):
            print (_char)

## 找到属于中文标点或者英文标点的字符
"""
{'.', '；', '『', '&', '^', ';', '“', '$', '%', ':', '”', '+', '』', '<', '，', '’', '!', ']', 
'·', '…', '（', '(', '_', '％', '"', ',', '：', '【', '》', '‘', '}', '】',
 '>', '[', '{', '）', '、', "'", '-', '。', '—', '＃', ')', '=', '！', '~', '？', 
 '/', '#', '?', '*', '@', '《'}
"""
def func_find_punc(sentence):
    puncs = []
    for _char in sentence:
        if is_chinese_punc(_char) or is_english_punc(_char):
            puncs.append(_char)
    return puncs

'''
， 80365
。 62812
？ 13566
！ 6222
. 4376
- 3917
） 2442
（ 2437
· 2346
、 2299
》 1525
《 1525
) 1300
( 1298
“ 904
” 900
： 828
— 788
[ 444
] 442
" 368
； 308
% 179
【 162
】 162
, 90
} 87
{ 85
/ 78
… 68
' 63
: 38
& 23
@ 21
# 14
? 12
; 12
_ 11
％ 10
* 9
+ 8
> 7
! 6
$ 5
＃ 4
= 4
^ 2
~ 2
< 2
‘ 2
’ 2
』 1
『 1
'''
def func_count_frequence(whole_puncs):
    count = {}
    for punc in whole_puncs:
        if punc not in count:
            count[punc] = 0
        count[punc] += 1
    
    puncs, nums = [], []
    for punc in count:
        puncs.append(punc)
        nums.append(count[punc])
    indexes = np.argsort(-np.array(nums))
    puncs = np.array(puncs)[indexes]
    nums  = np.array(nums)[indexes]

    for ii in range(len(puncs)):
        print (puncs[ii], nums[ii])

## 测试1：remove ! ? ...
def remove_emotional_punc(sentence):
    sentence = sentence.replace('？', '。')
    sentence = sentence.replace('！', '。')
    sentence = sentence.replace('?', '。')
    sentence = sentence.replace('!', '。')
    sentence = sentence.replace('…', '。')
    return sentence

def count_emotional_punc(sentence):
    gantanhao = sentence.count('！') + sentence.count('!')
    wenhao = sentence.count('？') + sentence.count('?')
    shenluehao = sentence.count('…')
    return gantanhao, wenhao, shenluehao

## 测试2：further remove sp
def remove_emotional_punc_space(sentence):
    sentence = remove_emotional_punc(sentence)
    sentence = sentence.replace('，', '')
    sentence = sentence.replace(',', '')
    return sentence

## 测试3：remvoe all func
def remove_all_punc(sentence):
    sentence_new = ""
    for _char in sentence:
        if is_chinese_punc(_char) or is_english_punc(_char):
            sentence_new += ''
        else: # 只保留不是标点符号的内容
            sentence_new += _char 
    return sentence_new

# ------ main function ------
def main_read_transcript_test():
    # store all puncs
    whole_puncs = []
    for dataset in config.PATH_TO_TRANSCRIPTIONS:
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        if trans_path is not None and os.path.exists(trans_path):
            chis = func_read_key_from_csv(trans_path, 'chinese')
            print (f'{dataset} -> sample number: {len(chis)}')
            for chi in chis:
                # chi = remove_emotional_punc(chi)
                # chi = remove_emotional_punc_space(chi)
                # chi = remove_all_punc(chi)
                puncs = func_find_punc(chi)
                whole_puncs.extend(puncs)
    print (set(whole_puncs))

    func_count_frequence(whole_puncs)


def transcript_conversion_one(old_trans, convert_type='case1'):
    names = func_read_key_from_csv(old_trans, 'name')
    chis  = func_read_key_from_csv(old_trans, 'chinese')
    engs  = func_read_key_from_csv(old_trans, 'english')
    print (f'process sample number: {len(names)}')

    assert convert_type in ['case1', 'case2', 'case3']
    for ii, chi in enumerate(chis):
        print (f'old: {chi}')
        if convert_type == 'case1':
            chis[ii] = remove_emotional_punc(chi)
        elif convert_type == 'case2':
            chis[ii] = remove_emotional_punc_space(chi)
        elif convert_type == 'case3':
            chis[ii] = remove_all_punc(chi)
        print (f'new: {chis[ii]}')
    save_trans = old_trans[:-4] + f'-{convert_type}.csv'
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chis[ii], engs[ii]]
    func_write_key_to_csv(save_trans, names, name2key, ['chinese', 'english'])


def transcript_conversion_multi():
    for dataset in config.PATH_TO_TRANSCRIPTIONS:
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        if trans_path is not None and os.path.exists(trans_path):
            transcript_conversion_one(trans_path, 'case1')
            transcript_conversion_one(trans_path, 'case2')
            transcript_conversion_one(trans_path, 'case3')


def punc_statistics():
    for dataset in ['MER2023', 'IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'MELD']:
        print (f'====== {dataset} ======')
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        label_path = config.PATH_TO_LABEL[dataset]

        # read name2chi
        names = func_read_key_from_csv(trans_path, 'name')
        chis  = func_read_key_from_csv(trans_path, 'chinese')
        print (f'process sample number: {len(names)}')
        name2chi = {}
        for ii in range(len(names)):
            name2chi[names[ii]] = chis[ii]
        
        # read process names
        process_names = []
        if dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
            corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
            corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
            corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
        elif dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
        elif dataset in ['MER2023']:
            corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
            corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
            corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
            corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
            process_names.extend(list(corpus.keys()))
        print (f'process_names: {len(process_names)}')

        whole_punc = []
        for name in process_names:
            gantanhao, wenhao, shenluehao = count_emotional_punc(name2chi[name])
            whole_punc.append(gantanhao+wenhao+shenluehao)
        print('avg punc', np.mean(whole_punc))


''' 统计感叹号，问号，省略号对于预测结果的影响 => [展示标点符号占比]
====== MER2023 ======
process sample number: 5030
process_names: 5030
avg punc 0.3025844930417495
====== IEMOCAPFour ======
process sample number: 10087
process_names: 5531
avg punc 0.3107937081902007
====== IEMOCAPSix ======
process sample number: 10087
process_names: 7433
avg punc 0.3227498990986143
====== CMUMOSI ======
process sample number: 2199
process_names: 2199
avg punc 0.01318781264211005
====== CMUMOSEI ======
process sample number: 22856
process_names: 22856
avg punc 0.05027126356317816
====== SIMS ======
process sample number: 2281
process_names: 2281
avg punc 0.07233669443226655
====== MELD ======
process sample number: 13708
process_names: 13708
avg punc 0.7092208929092501
====== SIMSv2 ======
process sample number: 4403
process_names: 4403
avg punc 0.07335907335907337
'''
                
           
from toolkit.globals import *
def generate_feature_extraction_cmds():
    whole_cmds = []
    for feature in featname_mapping:
        if feature in WHOLE_TEXT:
            for dataset in ['MER2023', 'MELD', 'SIMS', 'SIMSv2', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI']:
                for punc_case in ['case1', 'case2', 'case3']:
                    if feature == 'falcon-7b':
                        cmd = f"python extract_text_huggingface.py --dataset={dataset} --feature_level='UTTERANCE' --model_name={feature} --punc_case={punc_case} --gpu=-1"
                    else:
                        cmd = f"python extract_text_huggingface.py --dataset={dataset} --feature_level='UTTERANCE' --model_name={feature} --punc_case={punc_case} --gpu=0"
                    print (cmd)
                    whole_cmds.append(cmd)
    return whole_cmds


def generate_classification_cmds():
    whole_cmds = []
    for dataset in ['MER2023', 'MELD', 'SIMS', 'SIMSv2', 'IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI']:
        for feature_name in ['OPT-13B', 'ALBERT-small', 'Llama2-13B', 'PERT-base', 'RoBERTa-large', 'Baichuan-13B']:
            for punc_case in ['case1', 'case2', 'case3']:
                newname = featname_mapping_reverse[feature_name] + f'-punc{punc_case}-UTT'
                cmd = f"python -u main-release.py --model='attention' --feat_type='utt' --dataset={dataset} --audio_feature={newname} --text_feature={newname} --video_feature={newname} --save_root='./savedpunc' --gpu=0"
                print (cmd)
                whole_cmds.append(cmd)
    return whole_cmds

# python toolkit/utils/punctuation.py
if __name__ == '__main__':

    ## 生成待处理的字幕文件
    # transcript_conversion_multi()
    punc_statistics() # 找到为什么MELD会随着删除！？性能下降这么多

    ## 生成所有特征提取的脚本
    # generate_feature_extraction_cmds()

    ## 生成特征分类的命令行
    # generate_classification_cmds()

    