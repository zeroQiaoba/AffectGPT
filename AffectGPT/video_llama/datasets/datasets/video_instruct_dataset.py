import os
import copy
import json
import random
import pathlib
import pandas as pd
from PIL import Image
import decord
from decord import VideoReader
from typing import Dict, Optional, Sequence

import torch
import transformers
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.processors import transforms_video, AlproVideoTrainProcessor
from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.conversation.conversation_video import Conversation, SeparatorStyle


class Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, max_length=1024,
                       num_audio_query_token=8,
                       num_video_query_token=32, 
                       tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/'):

        data_path = pathlib.Path(ann_path)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         text_processor=text_processor,
                         vis_root=vis_root,
                         ann_path=ann_path,
                         max_length=max_length,
                         num_audio_query_token=num_audio_query_token,
                         num_video_query_token=num_video_query_token,
                         tokenizer_name=tokenizer_name)


    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        return full_video_fp

    '''
    inference format:
    ###Human: Close your eyes, open your ears and you imagine only based on the sound that <Audio><AudioHere></Audio>. \
    Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
    The subtitle of this video is <Subtitle>{subtitle}</Subtitle>. \
    Now answer my question based on what you have seen, heard, and subtitles. {user_message} ###Assistant:
    '''
    def __getitem__(self, index):
        num_retries = 10 # skip error videos
        for _ in range(num_retries):
            try:
                ## step1: read video features
                sample = self.annotation[index]
                video_path = self._get_video_path(sample)
                video, msg = load_video(
                    video_path=video_path,
                    n_frms = 8,
                    height = 224,
                    width  = 224,
                    sampling ="uniform",
                    return_msg = True
                )
                video = self.vis_processor.transform(video)
                
                ## step2: generate text_inputs and targets
                # text_input:[###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###]
                # label:     [-100.. -100, ....,                                      ...        xxx###-100...,        ...     xxx###]
                conversation_list = copy.deepcopy(sample['QA'])
                for ii in range(len(conversation_list)):
                    question = conversation_list[ii]['q']
                    answer   = conversation_list[ii]['a']
                    if ii == 0: # add additional inputs
                        question = "###Human:Close your ears, open your eyes and you see that <Video><ImageHere></Video>. " \
                                    + "Now answer my question based on what you have seen. " \
                                    + question + "###Assistant:"
                        replace_token = self.DEFAULT_IMAGE_PATCH_TOKEN * self.num_video_query_token
                        question = question.replace(self.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
                        answer = answer + "###"
                    else:
                        question = "Human:" + question + "###Assistant:"
                        answer = answer + "###"
                    conversation_list[ii]['q'] = question
                    conversation_list[ii]['a'] = answer

                text_input, label = [], []
                for ii in range(len(conversation_list)):
                    q_token_ids = self.to_token_ids(conversation_list[ii]['q'], self.max_length)
                    a_token_ids = self.to_token_ids(conversation_list[ii]['a'], self.max_length)
                    # generate (text_input, label)
                    text_input.append(q_token_ids)
                    text_input.append(a_token_ids)
                    label.append(torch.ones([len(q_token_ids)], dtype=q_token_ids.dtype) * self.IGNORE_INDEX)
                    label.append(a_token_ids)
                    # remove too long (text_input, label)
                    temp_input = torch.cat(text_input)
                    temp_label = torch.cat(label)
                    assert len(temp_input) == len(temp_label)
                    if len(temp_input) >= self.max_length:
                        text_input = text_input[:-2]
                        label = label[:-2]
                if len(text_input) == 0:
                    raise RuntimeError("too long text_input")
                text_input = torch.cat(text_input)
                label = torch.cat(label)
                if len(text_input) >= self.max_length:
                    raise RuntimeError("too long text_input")
            except:
                print(f"Failed to load or too long inputs: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        return {
            "image": video,
            "text_input": text_input,
            "label": label,
            'dataset': 'videochat_video',
        }