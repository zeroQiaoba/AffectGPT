import os
import copy
import json
import random
import pathlib
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import decord
from decord import VideoReader

import torch
from torch.utils.data.dataloader import default_collate

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.processors import transforms_video, AlproVideoTrainProcessor
from video_llama.conversation.conversation_video import Conversation,SeparatorStyle


class Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, max_length=1024,
                 num_video_query_token=32, num_audio_query_token=8, tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/'):

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

    
    def _get_image_path(self, sample):
        full_video_fp = os.path.join(self.vis_root,  sample['image'])
        return full_video_fp


    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                ## step1: read image features
                sample = self.annotation[index]
                image_path = self._get_image_path(sample)
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image) # [3, 224, 224]
                image = image.unsqueeze(dim=1) # [3, 1, 224, 224]

                ## step2: generate text_inputs and targets
                conversation_list = copy.deepcopy(sample['conversations'])
                assert len(conversation_list) % 2 == 0, f'must one for human and one for gpt'
                for ii in range(0, len(conversation_list), 2):
                    assert conversation_list[ii]['from'] == 'human'
                    assert conversation_list[ii+1]['from'] == 'gpt'
                    question = conversation_list[ii]['value']
                    answer   = conversation_list[ii+1]['value']
                    if ii == 0: # add additional inputs
                        question = question.replace('\n<image>', '')
                        question = question.replace('<image>\n', '')
                        question = "###Human:Close your ears, open your eyes and you see that <Image><ImageHere></Image>. " \
                                    + "Now answer my question based on what you have seen. " \
                                    + question + "###Assistant:"
                        replace_token = self.DEFAULT_IMAGE_PATCH_TOKEN * self.num_video_query_token
                        question = question.replace(self.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
                        answer = answer + "###"
                    else:
                        question = "Human:" + question + "###Assistant:"
                        answer = answer + "###"
                    conversation_list[ii]['value'] = question
                    conversation_list[ii+1]['value'] = answer

                text_input, label = [], []
                for ii in range(0, len(conversation_list), 2):
                    q_token_ids = self.to_token_ids(conversation_list[ii]['value'],   self.max_length)
                    a_token_ids = self.to_token_ids(conversation_list[ii+1]['value'], self.max_length)
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
                print(f"Failed to load or too long inputs: {image_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch image after {num_retries} retries.")
        
        return {
            "image": image,
            "text_input": text_input,
            "label": label,
            'dataset': 'llava_image',
        }
  