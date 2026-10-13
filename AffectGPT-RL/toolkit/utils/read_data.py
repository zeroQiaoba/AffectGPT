import os
import glob
import tqdm
import math
import pickle
import numpy as np
import multiprocessing

from ..globals import *
from .functions import *
from .read_files import *
from toolkit.models.modules.affectgpt.models.ImageBind.data import load_and_transform_audio_data


############################################################
# ------ for feat: feature_root+name -> (seqlen, featdim) ------
def func_read_one_feat(argv=None, feature_root=None, name=None, processor=None, model_name=None):
    feature_root, name, processor, model_name = argv

    # 路径可能的两个选项
    feature_path = os.path.join(feature_root, name+'.npy')
    feature_dir  = os.path.join(feature_root, name)

    feature = []
    if os.path.exists(feature_path): # audio/text => belong to speaker
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
        feature.append(single_feature)
    elif os.path.isdir(feature_dir):
        facenames = os.listdir(feature_dir) # 如果是文件夹，则依次读取文件夹内所有信息
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_dir, facename))
            feature.append(facefeat)
    else:
        raise Exception('feature path or dir do not exist!')

    # feature -> (seqlen, featdim)
    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 1:
        single_feature = single_feature[np.newaxis, :]
    return single_feature


# model_name：表示用的哪个预训练模型
# read multiple data [different datasets need different processors]
def func_read_multiprocess(feature_root, names, processor=None, read_type='feat', model_name=None):
    ## names => features
    params = []
    for name in names:
        params.append((feature_root, name, processor, model_name))

    # ------ debug ------
    # func_read_one_feat(params[0])
    # func_read_one_e2e_video(params[0])
    # func_read_one_e2e_audio(params[0])

    features = []
    with multiprocessing.Pool(processes=8) as pool:
        if read_type == 'feat':
            features = list(tqdm.tqdm(pool.imap(func_read_one_feat, params), total=len(params)))

    ## save (names, features)
    feature_shape = np.array(features[0]).shape
    feature_name = os.path.basename(feature_root)
    print (f'Input feature {feature_name} ===> dim is {feature_shape}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    return features, feature_shape[-1]


############################################################
# (seqlen, featdim) -> (dst_len, featdim)
def func_mapping_feature(feature, dst_len):
    featlen, featdim = feature.shape
    if featlen == dst_len:
        return feature
    elif featlen < dst_len:
        pad_feature = np.zeros((dst_len-featlen, featdim))
        feature = np.concatenate((pad_feature, feature), axis=0)
    else:
        if featlen // dst_len == featlen / dst_len:
            pad_len = 0
            pool_size = featlen // dst_len
        else:
            pad_len = dst_len - featlen % dst_len
            pool_size = featlen // dst_len + 1
        pad_feature = np.zeros((pad_len, featdim))
        feature = np.concatenate([pad_feature, feature]).reshape(dst_len, pool_size, featdim) # 相邻时刻特征取平均
        feature = np.mean(feature, axis=1)
    return feature

# sample-level
def align_to_utt(audios, texts, videos):
    for ii in range(len(audios)):
        audios[ii] = np.mean(audios[ii], axis=0)
        texts[ii]  = np.mean(texts[ii],  axis=0)
        videos[ii] = np.mean(videos[ii], axis=0)
    return audios, texts, videos

# sample-level: 每个模态的特征长度压缩到原来的scale倍
def feature_scale_compress(audios, texts, videos, scale_factor=1):
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature(audios[ii], math.ceil(len(audios[ii]) / scale_factor))
        texts[ii]  = func_mapping_feature(texts[ii],  math.ceil(len(texts[ii])  / scale_factor))
        videos[ii] = func_mapping_feature(videos[ii], math.ceil(len(videos[ii]) / scale_factor))
    return audios, texts, videos

# sample-level: 三种模态压缩到文本长度
def align_to_text(audios, texts, videos):
    for ii in range(len(audios)):
        dst_len = len(texts[ii])
        audios[ii] = func_mapping_feature(audios[ii], dst_len)
        texts[ii]  = func_mapping_feature(texts[ii],  dst_len)
        videos[ii] = func_mapping_feature(videos[ii], dst_len)
    return audios, texts, videos

# batch-level: generate batch
def pad_to_maxlen_pre_modality(audios, texts, videos):
    audio_maxlen = max([len(feature) for feature in audios])
    text_maxlen  = max([len(feature) for feature in texts ])
    video_maxlen = max([len(feature) for feature in videos])
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature(audios[ii], audio_maxlen)
        texts[ii]  = func_mapping_feature(texts[ii],  text_maxlen)
        videos[ii] = func_mapping_feature(videos[ii], video_maxlen)
    return audios, texts, videos
