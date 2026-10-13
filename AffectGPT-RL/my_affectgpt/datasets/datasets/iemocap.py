import os
import tqdm
import copy
import json
import random
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import decord
from decord import VideoReader

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from my_affectgpt.processors import transforms_video, AlproVideoTrainProcessor
from my_affectgpt.conversation.conversation_video import Conversation,SeparatorStyle
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio # split one function into two functions

import config

from toolkit.utils.functions import func_discrte_label_distribution


emos = ['happy', 'sad', 'neutral', 'anger']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo

# need the model to support all three inputs (audio, video, text)
# => train on the first 4 Sessions and test on the last Session; if run this way, the MERBench results should also be re-run
class IEMOCAPFour_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                       dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'IEMOCAPFour'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# manually specify all storage paths #################
        ## read train/test splits
        label_path = config.PATH_TO_LABEL[self.dataset]
        corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist()
        self.names = [name                for name in corpus]
        self.emos  = [idx2emo[corpus[name]['emo']] for name in corpus]

        train_idxs, test_idxs = self.split_indexes_using_session()
        train_names = [self.names[idx] for idx in train_idxs]
        train_emos  = [self.emos[idx]  for idx in train_idxs]
        self.test_names = [self.names[idx] for idx in test_idxs]
        self.test_emos  = [self.emos[idx]  for idx in test_idxs]
        
        name2subtitle = {}
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = row['name']
            subtitle = row['english']
            if pd.isna(subtitle): subtitle=""
            name2subtitle[name] = subtitle
        self.name2subtitle = name2subtitle
        
        self.annotation = []
        for ii, name in enumerate(train_names): # only for training data
            emotion = train_emos[ii]
            subtitle = name2subtitle[name]
            self.annotation.append({'name': name, 
                                    'subtitle': subtitle, 
                                    'onehot': emotion})
        self.label_type_candidates = ['onehot_w_candidates', 'onehot_wo_candidates']

        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= config.PATH_TO_RAW_FACE[self.dataset]

        self.candidate_labels = ",".join(set(train_emos))
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
    
    ## generate the index info needed for n-folder cross-validation
    def split_indexes_using_session(self):
        
        self.num_folder = 5

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(self.names):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == self.num_folder, f'Must split into five folder'

        ii = 4
        test_idxs = session_to_idx[ii]
        train_idxs = []
        for jj in range(self.num_folder):
            if jj != ii: train_idxs.extend(session_to_idx[jj]) # [0, 1, 2, 3]

        return train_idxs, test_idxs
    

    # two types of data formats may exist
    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.avi')
        return full_video_fp
    
    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp

    def _get_face_path(self, sample):
        full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
        return full_face_fp

    # for testing: if inference only supports image, randomly pick one frame as input
    def _get_image_path(self, sample):
        full_video_fp = self._get_video_path(sample)
        vr = VideoReader(uri=full_video_fp, height=224, width=224)

        # get index
        vlen = len(vr)
        index = random.randint(0, vlen-1)
        frame = vr[index]

        # read PIL.Image
        image = Image.fromarray(frame.numpy())
     
        return image
    
    # for inference
    def read_test_names(self):
        assert len(self.test_names) == 1241
        return self.test_names
    
    def get_test_name2gt(self):
        name2gt = {}
        for (name, emo) in zip(self.test_names, self.test_emos):
            name2gt[name] = emo
        return name2gt
    
    def get_emo2idx_idx2emo(self):
        return emo2idx, idx2emo
