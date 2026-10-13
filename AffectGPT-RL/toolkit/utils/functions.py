import os
import re
import cv2
import copy
import math
import tqdm
import glob
import shutil
import random
import argparse
import itertools
import torchaudio

from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

from toolkit.globals import *
from toolkit.utils.chatgpt import get_translate_eng2chi, get_translate_chi2eng
from toolkit.utils.read_files import func_write_key_to_csv
from toolkit.utils.read_files import func_read_key_from_csv
from toolkit.utils.read_files import func_split_list_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing

# 用于win上加速人脸提取 => 将name均匀分成两部分
def func_split_dataset_into_two_parts(dataset, split_num):
    whole_names = []
    video_root = config.PATH_TO_RAW_FACE_Win[dataset]
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        videoname = os.path.basename(video_path).rsplit('.', 1)[0]
        whole_names.append(videoname)
    
    store_root = './'
    func_split_list_data(whole_names, store_root, split_num, shuffle=True)


def func_avi_to_mp4(video_root):
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        if video_path.endswith('.mp4'): continue
        assert video_path.endswith('.avi'), f'unknown video type for {video_path}'

        try:
            save_path = video_path[:-4] + '.mp4'
            cmd = '%s -y -i \"%s\" -loglevel quiet -y \"%s\"' %(config.PATH_TO_FFMPEG, video_path, save_path)
            os.system(cmd)

            cmd = 'rm -rf \"%s\"' %(video_path)
            os.system(cmd)
        except:
            print (f'error videos: {video_path}')

# gain video paths
def func_gain_videopath(video_root, vid_name):
    video_path1 = os.path.join(video_root, vid_name + '.avi')
    video_path2 = os.path.join(video_root, vid_name + '.mp4')
    if os.path.exists(video_path1):
        return video_path1
    else:
        return video_path2

def func_gain_audiopath(video_root, vid_name):
    audio_path = os.path.join(video_root, vid_name + '.wav')
    return audio_path
    
# read name2trans
def func_gain_name2trans(trans_path):
    name2trans = {}
    names = func_read_key_from_csv(trans_path, 'name')
    chis  = func_read_key_from_csv(trans_path, 'chinese')
    for ii in range(len(names)):
        name2trans[names[ii]] = chis[ii]
    return name2trans

def func_read_audio_second(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if len(waveform.shape) == 2:
        duration = waveform.shape[1] / sr
    elif len(waveform.shape) == 1:
        duration = len(waveform) / sr
    else:
        ValueError('error waveform inputs')
    return duration

def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_decord_to_image(img):
    img = Image.fromarray(img)
    return img

################################################
## read face from openface outputs
def func_opencv_to_decord(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def func_video_to_face(vname):
    npy_path = os.path.join(config.PATH_TO_RAW_FACE[config.dataset], vname, vname+'.npy')
    assert os.path.exists(npy_path), f'error video has no face: {vname}'
    frames = np.load(npy_path)
    return frames

def load_video_from_npy(vname, n_frms=8, height=224, width=224, readtype='uniform', return_raw=False):

    frames = func_video_to_face(vname)
    vlen = len(frames)
    start, end = 0, vlen
    
    if readtype == 'all':
        indices = np.arange(start, end, 1).astype(int).tolist()
    elif readtype == 'uniform':
        n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif readtype == 'continuous': # for videomae
        ii = np.random.randint(start, max(start+1, end-n_frms))
        indices = np.arange(ii, min(end, ii+n_frms)).astype(int).tolist()
    elif readtype == 'continuous_polish': # for videomae [删除前后1s, 间隔4帧采样一次]
        start += 25 # 删除前后1s数据
        end   -= 25 
        ii = np.random.randint(start, max(start+1, end-n_frms*4))
        indices = np.linspace(ii, min(end, ii+n_frms*4), n_frms).astype(int).tolist()
 
    # whether compress into 'n_frms'
    if n_frms != 0:
        while len(indices) < n_frms:
            indices.append(indices[-1])
        indices = indices[:n_frms]
        assert len(indices) == n_frms, f'{indices}, {vlen}, {n_frms}'
    
    # indices -> images
    temp_frms = frames[indices] # [n_frms=8, 112, 112, 3]
    tgt_dim = (width, height)
    temp_frms = [cv2.resize(frm, tgt_dim)   for frm in temp_frms] # [n_frms=8, 224, 224, 3]
    temp_frms = [func_opencv_to_decord(frm) for frm in temp_frms] # [n_frms=8, 224, 224, 3]
    temp_frms = np.array(temp_frms) # [n_frms=8, 224, 224, 3]
     
    if return_raw is False:
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)
        return frms
    else:
        return temp_frms

## 临时测试，测试indexes的选择合理性 [ok]
# python toolkit/utils/functions.py debug_on_index_selection
def debug_on_index_selection(n_frms=16):
    for readtype in ['uniform', 'continuous']:
        for vlen in range(1, 100):
            print (f'{readtype}: vlen: {vlen}')
            start, end = 0, vlen
            
            if readtype == 'uniform':
                n_frms_update = min(n_frms, vlen)
                indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
            elif readtype == 'continuous':
                ii = np.random.randint(start, max(start+1, end-n_frms))
                indices = np.arange(ii, min(end, ii+n_frms)).astype(int).tolist()
            
            # get_batch -> T, H, W, C
            while len(indices) < n_frms:
                indices.append(indices[-1])

            assert len(indices) == n_frms
            print (indices)

################################################
# convert to map for storage
# def convert_omegaconf_to_map(config):
#     config_map = {}

#     for key in config: 
#         config_map[key] = config[key]
    
#     return config_map

# config -> args [只把config里存在，但是args中不存在或者为None的部分赋值]
def merge_args_config(args, config):
    args_dic = vars(args) # convert to map version
    for key in config:
        if key not in args_dic or args_dic[key] is None:
            args_dic[key] = config[key]
    args_new = argparse.Namespace(**args_dic) # change to namespace
    return args_new

# random select one param from model-tune.yaml
def func_random_select(config):
    for key in config:
        values = config[key]
        index = random.randint(0, len(values)-1)
        value = values[index]
        config[key] = value
    return config


## store into outputs
def func_update_storage(inputs, prefix, outputs):
    for key in inputs:
        val = inputs[key]
        # update key and value
        newkey = f'{prefix}_{key}'
        newval = val
        # store into outputs
        assert newkey not in outputs
        outputs[newkey] = newval


## 将openface压缩到.npy方便移动和操作
# python toolkit/utils/functions.py func_compress_openface_into_npy (face_root, save_root)
def func_compress_openface_into_npy(face_root, save_root):

    for face_dir in tqdm.tqdm(glob.glob(face_root + '/*')):
        frame_names = sorted(os.listdir(face_dir))

        frames = []
        for ii in range(len(frame_names)):
            frame_path = os.path.join(face_dir, frame_names[ii])
            frame = cv2.imread(frame_path)
            frames.append(frame)

        videoname = os.path.basename(face_dir)
        save_dir = os.path.join(save_root, videoname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, videoname+'.npy')
        np.save(save_path, frames)


# 将英文字幕翻译为中文
def func_translate_transcript(trans_path, save_path):

    names = func_read_key_from_csv(trans_path, 'name')
    engs  = func_read_key_from_csv(trans_path, 'english')
    print (f'whole sample number: {len(names)}')
    
    # translate eng2chi
    chis = []
    for eng in tqdm.tqdm(engs):
        chi = get_translate_eng2chi(eng, model='gpt-3.5-turbo-16k-0613')
        # chi = get_translate_eng2chi(eng, model='gpt-4-0613')
        chis.append(chi)

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chis[ii], engs[ii]]
    func_write_key_to_csv(save_path, names, name2key, ['chinese', 'english'])


# 适用于先 -> english.csv -> eng2chi.csv的情况
def func_translate_transcript_polish(trans_path, save_path, polish_path):
    names1 = func_read_key_from_csv(trans_path, 'name')
    engs1  = func_read_key_from_csv(trans_path, 'english')
    names2 = func_read_key_from_csv(save_path, 'name')
    engs2  = func_read_key_from_csv(save_path, 'english')
    chis2  = func_read_key_from_csv(save_path, 'chinese')
    print (f'sample number: {len(names1)}, {len(engs1)}')
    print (f'sample number: {len(names2)}, {len(engs2)}, {len(chis2)}')

    # 将没翻译的内容再次翻译一遍
    for ii, chi2 in tqdm.tqdm(enumerate(chis2)):
        name1, name2 = names1[ii], names2[ii]
        eng1,  eng2  = engs1[ii], engs2[ii]
        assert name1 == name2 and eng1 == eng2
        if len(chi2) == 0 and len(eng2) >= 2:
            print (f'error in {names1[ii]}')
            chi2_new = get_translate_eng2chi(eng2, model='gpt-3.5-turbo-16k-0613')
            chis2[ii] = chi2_new
        if chi2.find('\n')!=-1:
            print (f'error in {names1[ii]}')

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names1):
        name2key[name] = [chis2[ii], engs2[ii]]
    func_write_key_to_csv(polish_path, names2, name2key, ['chinese', 'english'])


# 适用于 直接是 eng2chi.csv 的情况
def func_translate_transcript_polish_merge(trans_path, polish_path):
    names = func_read_key_from_csv(trans_path, 'name')
    engs  = func_read_key_from_csv(trans_path, 'english')
    chis  = func_read_key_from_csv(trans_path, 'chinese')
    print (f'process sample number: {len(names)}')

    for ii, chi in tqdm.tqdm(enumerate(chis)):
        eng = engs[ii]
        if len(chi) == 0 and len(eng) != 0:
            print (f'error in {names[ii]}')
            chis[ii] = get_translate_eng2chi(eng, model='gpt-3.5-turbo-16k-0613')
        if len(eng) == 0 and len(chi) != 0:
            print (f'error in {names[ii]}')
            engs[ii] = get_translate_chi2eng(chi, model='gpt-3.5-turbo-16k-0613')

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chis[ii], engs[ii]]
    func_write_key_to_csv(polish_path, names, name2key, ['chinese', 'english'])


# 将audio从video中分割出来
def func_split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path) # linux
        # cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG_Win, video_path, audio_path) # windows
        os.system(cmd)


# 找到 split 出错的 audio data
# run -d toolkit/utils/functions.py func_find_false_audio 'dataset/meld-process/subvideo' 'dataset/meld-process/subaudio'
def func_find_false_audio(video_root, audio_root):
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(audio_root, videoname + '.wav')
        if not os.path.exists(audio_path):
            print (audio_path)

# feature -> tgt feature name
def func_name_conversion(audio, suffix):
    audio1 = featname_mapping_reverse[audio] + f'_{suffix}'
    audio2 = featname_mapping_reverse[audio] + f'-{suffix}'
    if os.path.exists(os.path.join(config.PATH_TO_FEATURES['SIMS'], audio1)):
        audio = audio1
    else:
        audio = audio2
    return audio

def func_check_feature_completeness(feature_root, intermerdia, suffix):
    # select features using (intermerdia, suffix)
    feature_names = os.listdir(feature_root)
    if suffix is not None:
        feature_names = [feature_name for feature_name in feature_names if feature_name.endswith(suffix)]
    if intermerdia is not None:
        feature_names = [feature_name for feature_name in feature_names if feature_name.find(intermerdia)!=-1]
    print (f'extracted feature numbers: {len(feature_names)}')

    for feature_name in feature_names:
        feature_dir = os.path.join(feature_root, feature_name)
        samples = glob.glob(feature_dir + '/*')
        sample_num = len(samples)
        if sample_num != 0:
            index = random.randint(0, sample_num-1)
            feature_shape = np.load(samples[index]).shape
        else:
            feature_shape = (0, 0)
        print (f'{feature_name} => shape: {feature_shape}  number: {sample_num}')


# suffix: 只统计以 suffix 为后缀的特征名
# target_dataset: 只统计目标数据集的结果
# intermerdia: 只统计内部包含 intermerdia 的特征
def check_feature_completeness(intermerdia=None, suffix=None, target_dataset=None):
    for dataset in config.PATH_TO_FEATURES:
        print (f'====== {dataset} ======')
        if target_dataset != dataset: continue
        feature_root = config.PATH_TO_FEATURES[dataset]
        if feature_root is not None and os.path.exists(feature_root):
            func_check_feature_completeness(feature_root, intermerdia, suffix)


def func_discrte_label_distribution(labels):
    print (f'sample number: {len(labels)}')
    print (f'label number: {len(set(labels))}')

    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1

    print ('label distribution')
    for label in sorted(label2count):
        print (label, ':', label2count[label])


#######################################
## add noise into audio
#######################################
import cv2
import wave
import array
def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params)
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

def func_add_noise_to_audio(argv=None, clean_path=None, noise_paths=None, snr=None, save_path=None):

    clean_path, noise_paths, snr, save_path = argv

    # read clean
    clean_wav = wave.open(clean_path, "r")
    clean_amp = cal_amp(clean_wav) # (采样点, )

    # read noise [ensure len(noise) > len(audio)]
    noise_amp = []
    while True:
        noise_path = noise_paths[random.randint(0, len(noise_paths)-1)]
        noise_wav = wave.open(noise_path, "r")
        noise_amp.extend(cal_amp(noise_wav))
        if len(noise_amp) > len(clean_amp): break
    start = random.randint(0, len(noise_amp) - len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    divided_noise_amp = np.array(divided_noise_amp)
    assert len(clean_amp) == len(divided_noise_amp)

    # clean + noise with snr
    clean_rms = cal_rms(clean_amp)
    noise_rms = cal_rms(divided_noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
    mixed_amp = clean_amp + adjusted_noise_amp
    
    # avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)

    save_waveform(save_path, clean_wav.getparams(), mixed_amp)


# choice_snrs = [0], [10], [20], [0, 10, 20] 这种混合snr
def add_noise_multiprocess(audio_root, choice_snrs):
    
    audio_paths = glob.glob(audio_root + '/*') # audio paths
    noise_paths = glob.glob(PATH_TO_NOISE + '/*') # noise path
    print (f'process audio: {len(audio_paths)}')
    print (f'candidate noises: {len(noise_paths)}')
    if len(choice_snrs) == 1:
        save_root = audio_root + f'_snr{choice_snrs[0]}'
    elif len(choice_snrs) == 3:
        save_root = audio_root + f'_snrmix'
    else:
        print ('Error: unsupported choice_snrs!!')
    if not os.path.exists(save_root): os.makedirs(save_root)

    params = []
    for audio_path in tqdm.tqdm(audio_paths):
        snr = choice_snrs[random.randint(0, len(choice_snrs)-1)]
        audio_name = os.path.basename(audio_path)
        save_path  = os.path.join(save_root, audio_name)
        params.append((audio_path, noise_paths, snr, save_path))

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_add_noise_to_audio, params), total=len(params)))

def whether_contains_chieng():
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
    # for dataset in ['SIMSv2']:
        print (f'process dataset: {dataset}')
        label_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        chis = func_read_key_from_csv(label_path, 'chinese')
        engs = func_read_key_from_csv(label_path, 'english')
        print (f'sample number: {len(chis)}  {len(engs)}')
        print (f'eng: {engs[100]}')
        print (f'chi: {chis[100]}')


def generate_cmds_for_feature_extraction():
    whole_cmds = []
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        for model in ['Llama-2-13b-hf', 'bloom-7b1', 'Baichuan-13B-Base', 'falcon-7b']:
            cmd = f"python extract_text_huggingface.py --dataset={dataset} --feature_level='UTTERANCE' --model_name={model} --language='english' --gpu=0"
            whole_cmds.append(cmd)
    indices = np.arange(len(whole_cmds))
    random.shuffle(indices)
    whole_cmds = np.array(whole_cmds)[indices]
    print (f'whole cmds numbers: {len(whole_cmds)}')
    for cmd in whole_cmds: print (cmd)
    return whole_cmds

def func_none_or_str(value):
    if value.lower() == "none":
        return None
    return value

def func_label_distribution(labels):
    label2count = {}
    for label in labels:
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1
    return label2count

def func_major_labels(labels):
    label2count = func_label_distribution(labels)
    maxcount = max([label2count[label] for label in label2count])

    maxlabels = []
    for label in label2count:
        if label2count[label] == maxcount:
            maxlabels.append(label)
        
    return maxlabels

def func_majoremo_majorcount(labels):
    label2count = func_label_distribution(labels)
    maxcount = max([label2count[label] for label in label2count])

    maxlabels = []
    for label in label2count:
        if label2count[label] == maxcount:
            return maxcount, label


# labels, preds: should be integrate
# target_names: idx -> label names
def func_plot_confusion_matrix(labels, preds, target_names, save_path, fontsize=12, cmap=plt.cm.Blues):

    # gain cm
    assert len(set(labels)) == len(target_names)
    cm = confusion_matrix(labels, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot cm
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(target_names))
    # plt.xticks(tick_marks, target_names, rotation=45, fontsize=fontsize)
    plt.xticks(tick_marks, target_names, rotation=20, fontsize=fontsize)
    plt.yticks(tick_marks, target_names, fontsize=fontsize)
    # plt.ylabel('True Label', fontsize=fontsize)
    # plt.xlabel('Predicted Label', fontsize=fontsize)
    plt.ylabel('RGB Image Predictions',  fontsize=fontsize+4)
    plt.xlabel('Grayscale Image Predictions', fontsize=fontsize+4)
    plt.tight_layout()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(f'{cm[i, j]:.2f}'),
                 horizontalalignment="center",
                 fontsize = fontsize,
                 color="white" if cm[i, j] > thresh else "black")
    plt.savefig(save_path, format='png')


def func_find_refused_samples(label_path, gpt4v_path, image_root, store_root):
    all_names = func_read_key_from_csv(label_path, 'name')
    # all_names = [f'{name}.jpg' for name in all_names]
    print (f'all samples: {len(all_names)}')
    passed_names = func_read_key_from_csv(gpt4v_path, 'name')
    # passed_names = [f'{name}.jpg' for name in passed_names]
    print (f'passed samples: {len(passed_names)}')

    passed_root = os.path.join(store_root, 'passed')
    no_passed_root = os.path.join(store_root, 'no_passed')
    if not os.path.exists(passed_root): os.makedirs(passed_root)
    if not os.path.exists(no_passed_root): os.makedirs(no_passed_root)

    for name in all_names:
        input_path = f'{image_root}/{name}'
        if name in passed_names:
            save_path = f'{passed_root}/{name}'
        else:
            save_path = f'{no_passed_root}/{name}'
        if os.path.isfile(input_path):
            shutil.copy(input_path, save_path)
        elif os.path.isdir(input_path):
            shutil.copytree(input_path, save_path)


# 整体上，这个距离的计算并不准确，要不还是转换到chatgpt上吧
def func_calculate_emotion_similarity():

    from gensim.models import KeyedVectors # 这个距离计算不准
    # model = KeyedVectors.load_word2vec_format('/share/home/lianzheng/tools/word2vec/GoogleNews-vectors-negative300.bin', binary=True) 
    # model = KeyedVectors.load_word2vec_format('/share/home/lianzheng/tools/word2vec/tencent-ailab-embedding-en-d100-v0.1.0', binary=False) 
    model = KeyedVectors.load_word2vec_format('/share/home/lianzheng/tools/word2vec/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt', binary=False) # 感觉中文的这个还靠谱点。。。可以挑一个测测看吧

    ## 将句子换成单词呢 [完全不行呢]
    emos1 = ['happy', 'pleasure', 'happiness', 'sad']
    emos2 = ['happy', 'pleasure', 'happiness', 'sad']

    for i in range(len(emos1)):
        for j in range(len(emos2)):
            score = model.wv.similarity(emos1[i], emos2[j])
            print("{} \t\t {} \t\t Score: {:.4f}".format(emos1[i], emos2[j], score))

# "['轻松', '愉快', '幽默', '自嘲']" => ['轻松', '愉快', '幽默', '自嘲']
def string_to_list(str):

    if type(str) == np.ndarray:
        str = str.tolist()

    # 如果本身就是list了，那么就不需要其他操作了
    if isinstance(str, list):
        return str
    
    if str == '':
        str = []
    elif pd.isna(str):
        str = []
    else:
        if str[0] == '[':  str = str[1:]
        if str[-1] == ']': str = str[:-1]
        str = [item.strip() for item in re.split('[\'\",]', str) if item.strip() not in ['', ',']]
    return str

# "[['轻松', '愉快'], ['幽默', '自嘲'], ['高度焦虑']]" 
# => [['轻松', '愉快'], ['幽默', '自嘲'], ['高度焦虑']]
def listlist_to_list(str):
    results = []

    multi_lists = [item for item in re.split(r"[\[\]]", str) if item.strip() not in ['', ',']]
    for one_list in multi_lists:
        one_list = [item for item in re.split('[\'\"]', one_list) if item.strip() not in ['', ',']]
        results.append(one_list)
    
    return results

def calculate_pcc(y_true, y_pred):
    pcc = np.corrcoef(y_true, y_pred)[0, 1]
    return pcc

import pandas as pd
def calculate_ccc(y_true, y_pred):
    
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)

    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator

def func_normalize_to_0_to_1(values):
    minval = np.min(values)
    maxval = np.max(values)
    values = (values - minval) / (maxval - minval)
    return values


def split_list_into_batch(items, split_num=None, batchsize=None):
    
    assert (split_num is not None) or (batchsize is not None)

    if split_num is None:
        split_num = math.ceil(len(items)/batchsize)

    # split infos into subset
    batches = []
    # items = np.array(items)
    each_split = math.ceil(len(items)/split_num)
    for ii in range(split_num):
        batch = items[ii*each_split:(ii+1)*each_split]
        if len(batch)!=0: 
            batches.append(batch)

    return batches


# 按照 key_lists，更新 map_variable，存储值为 values
def create_nested_dict(keys, value):
    nested_dict = {}
    d = nested_dict
    for key in keys[:-1]:  # 遍历除了最后一个键之外的所有键
        d = d.setdefault(key, {})  # 创建嵌套字典
    d[keys[-1]] = value  # 将值赋给最后一个键
    return nested_dict


def merge_dicts(base_dict, new_dict):
    # 合并两个嵌套字典
    for key, value in new_dict.items():
        if key in base_dict:
            merge_dicts(base_dict[key], value)  # 递归合并
        else:
            base_dict[key] = value  # 直接添加新的键值对


## 要求 base_dict 和 new_dict 的结构是一致的
def add_dicts(base_dict, new_dict):
    # 将两个嵌套字典，对应元素相加
    for key, value in new_dict.items():
        assert key in base_dict
        if isinstance(value, dict):
            add_dicts(base_dict[key], value) # 递归合并
        elif isinstance(value, int):
            base_dict[key] += value # 数值合并
        elif isinstance(value, list):
            base_dict[key] = [a + b for a, b in zip(base_dict[key], value)] # 两个 list 逐元素相加

## 将 base_dict 中每个元素除以一个 div
def div_dicts(base_dict, div):
    for key, value in base_dict.items():
        if isinstance(value, dict):
            div_dicts(base_dict[key], div) # 递归合并
        elif isinstance(value, int):
            base_dict[key] /= div
        elif isinstance(value, list):
            base_dict[key] = [item/div for item in base_dict[key]]

## 给出了一组结构一致的map，计算他们对应元素的平均值
'''
39.23127574617797, 36.22835048869017, 42.824557105116256
38.58412466133963, 35.70549439260656, 42.02193050
38.9077002037588, 35.96692244064836, 42.42324380677136
1. dict_list不变 => [check ok]
2. values 是对的 => [check ok]
'''
def main_mean_dicts(dict_list):
    base_dict = copy.deepcopy(dict_list[0])
    for new_dict in dict_list[1:]:
        add_dicts(base_dict, new_dict)
    div_dicts(base_dict, div=len(dict_list))
    return base_dict


## 要求 base_dict 和 new_dict 的结构是一致的，将他们stack起来
def concat_dicts(base_dict, new_dict):
    # 将两个嵌套字典，对应元素相加
    for key, value in new_dict.items():
        assert key in base_dict
        if isinstance(value, dict):
            concat_dicts(base_dict[key], value) # 递归合并
        elif isinstance(value, int):
            if isinstance(base_dict[key], int):
                base_dict[key] = [base_dict[key], value]
            elif isinstance(base_dict[key], list):
                base_dict[key].append(value)
        elif isinstance(value, list):
            base_dict[key] = np.vstack([base_dict[key], value])

def std_dicts(base_dict):
    for key, value in base_dict.items():
        if isinstance(value, dict):
            std_dicts(base_dict[key]) # 递归合并
        elif np.array(value).ndim == 1: # 一维数组
            base_dict[key] = np.std(value)
        elif np.array(value).ndim == 2: # 二维数组
            base_dict[key] = np.std(value, axis=0)

# 将 base_dict 里面的所有值设置为0
def zero_dicts(base_dict):
    for key, value in base_dict.items():
        if isinstance(value, dict):
            zero_dicts(base_dict[key]) 
        else:
            base_dict[key] = 0

## 给出了一组结构一致的map，计算他们对应元素的方差
'''
run1: 39.23127574617797, 36.22835048869017, 42.824557105116256
run2: 38.58412466133963, 35.70549439260656, 42.02193050
std:  0.32357554,        0.26142805,        0.4013133
1. dict_list不变 => [check ok]
2. values 是对的 => [check ok]
'''
def main_std_dicts(dict_list):
    if len(dict_list) == 1:
        base_dict = copy.deepcopy(dict_list[0])
        zero_dicts(base_dict)
    else:
        base_dict = copy.deepcopy(dict_list[0])
        for new_dict in dict_list[1:]:
            concat_dicts(base_dict, new_dict)
        std_dicts(base_dict)
    return base_dict


## 曲线平滑函数
def func_smooth(results, winnum=50):
        half_win = int(winnum/2)
        smooth_result = []
        smooth_result.extend(results[:half_win]) # [0, 24]
        for epoch in range(half_win, len(results)-half_win):
            left_margin  = max(0, epoch-half_win)
            right_margin = min(len(results), epoch+half_win)
            win_select = results[left_margin:right_margin]
            smooth_result.append(np.mean(win_select))
        smooth_result.extend(results[-half_win:])
        return smooth_result

## 是不是 中文字符
def is_chinese_character(_char):
    if '\u4e00' <= _char <= '\u9fff':
        return True
    else:
        return False

## 判断文本中，是否有中文字符
def text_has_chinese_char(text):
    for _char in text:
        if is_chinese_character(_char):
            return True
    return False

##########################################################################
## 多进程视频转换
def func_avi_to_webm(argv=None, video_path=None, save_path=None):
    if argv != None:
        video_path, save_path = argv

    cmd = "%s -loglevel quiet -y -i %s %s" %(config.PATH_TO_FFMPEG, video_path, save_path)
    os.system(cmd)
    
def main_avi_to_webm_multiprocess(video_root, save_root):

    if not os.path.exists(save_root): 
        os.makedirs(save_root)
    
    params = []
    for video_path in glob.glob(video_root + '/*'):
        video_name = os.path.basename(video_path)[:-4]
        save_path = os.path.join(save_root, video_name+'.webm')
        params.append((video_path, save_path))

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_avi_to_webm, params), total=len(params)))
##########################################################################

# 找到 top 80% 对应的阈值是多少
def func_top_threshold(scores, top=0.8):
    scores = np.sort(scores)
    return scores[int(len(scores)*top)]


if __name__ == '__main__':
    import fire
    fire.Fire()

    # # ## add noise into audios
    # for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
    #     for choice_snrs in [[5], [10], [15], [5, 10, 15]]:
    #         audio_root = config.PATH_TO_RAW_AUDIO[dataset]
    #         add_noise_multiprocess(audio_root, choice_snrs)

    ## check ok, then you can extract english features
    # whether_contains_chieng()
    # generate_cmds_for_feature_extraction() # 打印特征提取脚本

    ## test feature completeness [check ok]
    # check_feature_completeness(intermerdia='langeng', suffix='UTT')
    # check_feature_completeness(intermerdia='noise', suffix='UTT')

    # func_calculate_emotion_similarity()
