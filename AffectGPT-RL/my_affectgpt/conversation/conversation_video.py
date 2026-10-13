import re
import copy
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from PIL import Image
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from my_affectgpt.common.registry import registry
from my_affectgpt.processors import Blip2ImageEvalProcessor
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
from my_affectgpt.datasets.builders.image_text_pair_builder import *
import config

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


######################################################
# ============== (only for inference) ============== #
######################################################
class Chat:
    def __init__(self, model, model_cfg, device='cuda:0'):
        self.device = device
        self.model = model
        self.tokenizer = model.llama_tokenizer

        # force transformers to use the legacy cache format (no conversion to DynamicCache)
        if hasattr(model.llama_model, 'generation_config'):
            model.llama_model.generation_config.return_legacy_cache = True

        # use a more general format; a standalone "#" and an "a#" in text encode to different tokens
        id_tre_jin = self.tokenizer('###', add_special_tokens=False)['input_ids'][0] # 835
        id_two_jin = self.tokenizer('a##', add_special_tokens=False)['input_ids'][1] # 2277
        id_one_jin = self.tokenizer('a#', add_special_tokens=False)['input_ids'][1]  # 29937
        stop_words_ids = [torch.tensor([self.tokenizer.eos_token_id]).to(self.device),
                          torch.tensor([id_tre_jin]).to(self.device),
                          torch.tensor([id_two_jin, id_one_jin]).to(self.device),
                          torch.tensor([id_one_jin, id_two_jin]).to(self.device)] # three id for "### / # ## "
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # set other params
        self.num_video_query_token = model_cfg.num_video_query_token
        self.num_audio_query_token = model_cfg.num_audio_query_token
        self.num_multi_query_token = model_cfg.num_multi_query_token
        self.num_image_query_token = model_cfg.num_image_query_token


    def to_token_ids(self, text, max_length):
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=max_length,
                                   truncation=True,
                                   add_special_tokens=False).input_ids[0]
        return input_ids

    def replace_token_for_multimodal(self, prompt):
        replace_token = config.DEFAULT_FRAME_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FRAME_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_FACE_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FACE_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_AUDIO_PATCH_TOKEN * self.num_audio_query_token
        prompt = prompt.replace(config.DEFAULT_AUDIO_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_MULTI_PATCH_TOKEN * self.num_multi_query_token
        prompt = prompt.replace(config.DEFAULT_MULTI_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_IMAGE_PATCH_TOKEN * self.num_image_query_token
        prompt = prompt.replace(config.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
        return prompt
   
    def postprocess_audio(self, sample_data):
        if sample_data['audio'] is None:
            return None, None
        
        audio = sample_data['audio'].unsqueeze(0).to(self.device)
        raw_audio = sample_data['raw_audio'].unsqueeze(0).to(self.device)
        audio_hiddens, audio_llms = self.model.encode_audio_merge(audio, raw_audio)
        return audio_hiddens, audio_llms

    def postprocess_face(self, sample_data):
        if sample_data['face'] is None:
            return None, None
        
        face = sample_data['face'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_face = sample_data['raw_face'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        face_hiddens, face_llms = self.model.encode_video_merge(face, raw_face)
        return face_hiddens, face_llms
    
    def postprocess_frame(self, sample_data):
        if sample_data['frame'] is None:
            return None, None
        
        video = sample_data['frame'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_video = sample_data['raw_frame'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        frame_hiddens, frame_llms = self.model.encode_video_merge(video, raw_video)
        return frame_hiddens, frame_llms

    def postprocess_image(self, sample_data):
        if sample_data['image'] is None:
            return None, None
        
        image = sample_data['image'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_image = sample_data['raw_image'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        image_hiddens, image_llms = self.model.encode_image_merge(image, raw_image)
        return image_hiddens, image_llms


    def postprocess_multi(self, video_hiddens, audio_hiddens):
        if video_hiddens is None or audio_hiddens is None:
            return None, None

        multi_hiddens, multi_llms = self.model.encode_multi_merge(video_hiddens, audio_hiddens)
        return multi_hiddens, multi_llms

    
    # the whole process simulates inference => tries to read/write exactly as in training
    # mainly because samples come one at a time here, not as a batch, so it simulates processing one sample at a time
    def answer_sample(self, prompt, img_list, num_beams=1, temperature=1.0, do_sample=True,  top_p=0.9,
                    max_new_tokens=1000, min_length=1, max_length=2000, repetition_penalty=1.0, length_penalty=1.0):
        
        
        IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]

        ###### step1: => (input_id, attention_mask) 
        ## replace and add
        prompt = self.replace_token_for_multimodal(prompt)
        input_id = self.to_token_ids(prompt, max_length)
        print (prompt)
        ## => I tried <bos> + <eos> and the output was empty; maybe the <bos> token added in front triggers some stop condition?

        ## length limits
        current_max_len = len(input_id) + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        input_id = input_id[begin_idx:]
        attention_mask=input_id.ne(self.tokenizer.pad_token_id).to(self.device)

        ###### step2: (input_ids) => (inputs_embeds)
        temp_input_id = copy.deepcopy(input_id).to(self.device)
        temp_input_id[temp_input_id == FRAME_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == FACE_PATCH_TOKEN_ID]  = 0
        temp_input_id[temp_input_id == AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == MULTI_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == IMAGE_PATCH_TOKEN_ID] = 0
        cur_input_embeds = self.model.llama_model.model.model.embed_tokens(temp_input_id)
        cur_input_ids = input_id
        
        # replace <ImageHere>, <AudioHere>, <FrameHere>, <FaceHere> with features
        cur_idx = 0
        for (patch_token_id, query_token_number, embeds) in [(FRAME_PATCH_TOKEN_ID, self.num_video_query_token, img_list['frame']),
                                                             (FACE_PATCH_TOKEN_ID,  self.num_video_query_token, img_list['face']),
                                                             (AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, img_list['audio']),
                                                             (MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, img_list['multi']),
                                                             (IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, img_list['image']),
                                                            ]:
            if (cur_input_ids == patch_token_id).sum() != 0:
                assert embeds is not None, f'Some input info is missing.'
                cur_features = embeds[cur_idx]
                if (cur_input_ids == patch_token_id).sum() != query_token_number:
                    raise ValueError("The number of audio patch tokens should be the same as the number of audio patches.")
                masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+query_token_number, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], 
                                            cur_features, 
                                            cur_input_embeds[mask_index_start+query_token_number:]), dim=0)
                    
        cur_input_embeds = cur_input_embeds.unsqueeze(0) 
        attention_mask = attention_mask.unsqueeze(0) 
        ###### step3: (inputs_embeds, attention_masks) => response
        ###############################################################
        ## when I switched to Baichuan2 this inference part had problems; most likely an issue from the transformers version
        with self.model.maybe_autocast(): # automatic mixed precision, no extra manual tuning needed
            outputs = self.model.llama_model.generate(
                inputs_embeds=cur_input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )

        ###### step4: convert to batch samples 
        # maybe <bos> aaa <stop token> bbb <eos>
        response = self.tokenizer.decode(outputs[0], add_special_tokens=False)
        if response.find(self.tokenizer.bos_token) != -1:
            response = response.split(self.tokenizer.bos_token)[1]
        if response.find(self.tokenizer.eos_token) != -1:
            response = response.split(self.tokenizer.eos_token)[0]
        response = response.rsplit('###', 1)[0] # split from stop tokens '###'
        response = response.split('Assistant:')[-1].strip()
        return response
