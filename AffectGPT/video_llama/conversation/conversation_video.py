import re
import os
import sys
import time
import argparse
import dataclasses
import numpy as np
from PIL import Image
from enum import auto, Enum
from typing import List, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from video_llama.common.registry import registry
from video_llama.processors import Blip2ImageEvalProcessor
from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor # video transformer
        self.image_vis_processor = Blip2ImageEvalProcessor() # image transformer
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)] # three id for "# / ## / ###"
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


    # only the first round conversation add input
    def ask(self, user_message, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1] or '</Audio>' in conv.messages[-1][1]):
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], user_message])
        else:
            conv.append_message(conv.roles[0], user_message)

    
    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1.0, temperature=1.0, max_length=2000):
        
        conv.append_message(conv.roles[1], None)    # conv[-1] = ['Assistant', None]
        embs = self.get_context_emb(conv, img_list) # merge into llama-based embeds

        # max_new_tokens: max length for response
        # target: make len(input)+len(response) < max_length
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        
        # even with same 'embs', llama also contains randomness, 
        # maybe figure out the effect of different params
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0: output_token = output_token[1:] # 0: <unk>
        if output_token[0] == 1: output_token = output_token[1:] # 1: bos or <s>
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0] # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip() 
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()
    

    def upload_video(self, video_path, conv, img_list, subtitle):
        if isinstance(video_path, str):
            # video: [3, 8, 224, 224]
            video, _ = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", 
                return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        else:
            raise NotImplementedError
        
        # audio_flag: for special cases where one video contains no audio
        try:
            audio_flag = 1
            audio = load_and_transform_audio_data([video_path], "cpu", clips_per_video=8) # [1, 8, 1, 128, 204]
            audio = audio.to(self.device)
        except :
            print('no audio is found')
            audio_flag = 0
        finally:
            if audio_flag == 1:
                audio_emb,_  = self.model.encode_audioQformer(audio)        # [1, 8, 4096]
                image_emb, _ = self.model.encode_videoQformer_visual(video) # [1, 32, 4096]
                img_list.append(audio_emb)
                img_list.append(image_emb)
                conv.system = ""
                conv.append_message(conv.roles[0], f"Close your eyes, open your ears and you imagine only based on the sound that: <Audio><AudioHere></Audio>. \
                Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
                The subtitle of this video is <Subtitle>{subtitle}</Subtitle>. \
                Now answer my question based on what you have seen, heard, and given subtitles.")
            else: # for video which contains no audio info
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                img_list.append(image_emb)
                conv.append_message(conv.roles[0], f"Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
                The subtitle of this video is <Subtitle>{subtitle}</Subtitle>. \
                Now answer my question based on what you have seen, and given subtitles.")
            return "Received."


    # def upload_video_without_audio(self, video_path, conv, img_list):
    #     msg = ""
    #     if isinstance(video_path, str):
    #         ext = os.path.splitext(video_path)[-1].lower()
    #         print(video_path)
    #         video, msg = load_video(
    #             video_path=video_path,
    #             n_frms=8,
    #             height=224,
    #             width=224,
    #             sampling ="uniform", return_msg = True
    #         )
    #         video = self.vis_processor.transform(video)
    #         video = video.unsqueeze(0).to(self.device)
    #     else:
    #         raise NotImplementedError
    #     image_emb, _ = self.model.encode_videoQformer_visual(video)
    #     img_list.append(image_emb)
    #     conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
    #     return "Received."


    # def upload_img(self, image, conv, img_list):
    #     msg = ""
    #     if isinstance(image, str):  # is a image path
    #         raw_image = Image.open(image).convert('RGB') # 增加一个时间维度
    #         image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
    #     elif isinstance(image, Image.Image):
    #         raw_image = image
    #         image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
    #     elif isinstance(image, torch.Tensor):
    #         if len(image.shape) == 3:
    #             image = image.unsqueeze(0)
    #         image = image.to(self.device)
    #     else:
    #         raise NotImplementedError

    #     image_emb, _ = self.model.encode_videoQformer_visual(image)
    #     img_list.append(image_emb)
    #     conv.append_message(conv.roles[0], "<Image><ImageHere></Image> "+ msg)
    #     return "Received."


    ## merge prompt and img_list into llama-level embs
    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = re.split('<ImageHere>|<AudioHere>', prompt)
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of placeholders and inputs."
        seg_tokens = [
            self.model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=(i==0)).to(self.device).input_ids # add <s> in the beginning
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs