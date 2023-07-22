"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import tqdm
import copy
import json
import random
import pathlib
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence, Iterable

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

class BaseDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, 
                 max_length=1024,
                 num_audio_query_token=8, 
                 num_video_query_token=32, 
                 tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/'):
        
        self.vis_root = vis_root
        self.ann_path = ann_path
        self.max_length = max_length
        self.vis_processor  = vis_processor
        self.text_processor = text_processor
        self.num_video_query_token = num_video_query_token
        self.num_audio_query_token = num_audio_query_token

        self.IGNORE_INDEX = -100
        self.DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'

        # 这里token的设置和video_llama.py中的一致
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens([self.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens([self.DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[self.DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[self.DEFAULT_AUDIO_PATCH_TOKEN]
        
        ####################################
        ## debug1: for all datasets
        sample1 = self.__getitem__(0)
        sample2 = self.__getitem__(0)
        sample3 = self.__getitem__(-1)
        self.func_visualize_samples(sample1)
        self.func_visualize_samples(sample2)
        self.func_visualize_samples(sample3)
        samples = [sample1, sample2, sample3]
        self.collater(samples)

        ## debug2: for all datasets (whether contains errors)
        # for index in tqdm.tqdm(range(len(self))):
        #     self.__getitem__(index)
        ####################################

    
    def __len__(self):
        return len(self.annotation)
    

    def func_visualize_samples(self, sample):
        text_input = copy.deepcopy(sample['text_input'])
        input_convert = self.tokenizer.decode(text_input)
        print (input_convert)

        label = copy.deepcopy(sample['label'])
        label[label==self.IGNORE_INDEX] = 2
        output_convert = self.tokenizer.decode(label)
        print (output_convert)
    
    
    def to_token_ids(self, text, max_length):
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=max_length,
                                   truncation=True,
                                   add_special_tokens=False).input_ids[0]
        return input_ids


    # post-process for batch
    def collater(self, instances):
        '''
        data_dict: input_ids:[###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###]
                   labels:   [-100...###, -100, ....,                                 ...           xxx###-100...,        ...     xxx###]

        data_dict: input_ids:[<s>###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###, 2,    ...]
                   labels:   [-100...###, -100, ....,                                 ...           xxx###-100...,        ...     xxx###, -100, ...]
                   images:   [bs=3, c=3, 224, 224]
        '''
        labels = []
        input_ids = []
        for instance in instances:
            label = instance['label']
            input_id = instance['text_input']
            label = torch.cat([torch.ones([1], dtype=input_id.dtype) * -100, label])    # add bos
            input_id = torch.cat([torch.ones([1], dtype=input_id.dtype),     input_id]) # add bos
            labels.append(label)
            input_ids.append(input_id)

        # pad bacth input into the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=True, 
                                                    padding_value=self.tokenizer.pad_token_id)
        labels    = torch.nn.utils.rnn.pad_sequence(labels,    
                                                    batch_first=True, 
                                                    padding_value=self.IGNORE_INDEX)
        batch = dict(
            labels=labels,
            input_ids=input_ids,
            attention_masks=input_ids.ne(self.tokenizer.pad_token_id), # mask padded input
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'audio' in instances[0]:
            audios = [instance['audio'] for instance in instances]
            if all(x is not None and x.shape == audios[0].shape for x in audios):
                batch['audios'] = torch.stack(audios)
            else:
                batch['audios'] = audios
        
        batch['dataset'] = instances[0]['dataset']
        return batch
    

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
