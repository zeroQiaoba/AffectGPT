import os
import copy
import json
import random
import pathlib
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from video_llama.datasets.datasets.base_dataset import BaseDataset


class CCSBUAlignDataset(BaseDataset):
    def __init__(self,  vis_processor, text_processor, vis_root, ann_path, max_length=1024,
                        num_audio_query_token=8,
                        num_video_query_token=32, 
                        tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/'):
        
        self.prompts = ["Describe this image in detail.",
                        "Take a look at this image and describe what you notice.",
                        "Please provide a detailed description of the picture.",
                        "Could you describe the contents of this image for me?",
                        ]

        self.annotation = json.load(open(ann_path, "r"))['annotations']

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
        full_video_fp = os.path.join(self.vis_root,  sample['image_id'] + '.jpg')
        return full_video_fp
    

    def _random_prompts(self):
        index = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[index]
        return prompt


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
                prompt =  self._random_prompts()
                prompt = f"###Human: Close your ears, open your eyes and you see that <Image><ImageHere></Image>. "  \
                         + f"Now answer my question based on what you have seen. {prompt} ###Assistant:"
                replace_token = self.DEFAULT_IMAGE_PATCH_TOKEN * self.num_video_query_token
                prompt = prompt.replace(self.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
                prompt_id = self.to_token_ids(prompt, self.max_length)
                
                ## step3: process for target
                reason = sample['caption'] + '###'
                target_id = self.to_token_ids(reason, self.max_length)

                text_input = torch.cat([prompt_id, target_id])
                label = torch.cat([torch.ones([len(prompt_id)], dtype=text_input.dtype) * -100, target_id])
                assert len(text_input) == len(label)
                if len(text_input) > self.max_length:
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
            'dataset': 'cc_image'
        }
