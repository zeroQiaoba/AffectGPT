"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import my_affectgpt.common.dist_utils as dist_utils
from my_affectgpt.common.dist_utils import download_cached_file
from my_affectgpt.common.utils import is_url
from my_affectgpt.common.logger import MetricLogger
from my_affectgpt.models.base_model import BaseModel
from my_affectgpt.models.Qformer import BertConfig, BertLMHeadModel
from my_affectgpt.models.eva_vit import create_eva_vit_g
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, AutoImageProcessor, Wav2Vec2FeatureExtractor
import config
from my_affectgpt.models.blip2 import Blip2Base, disabled_train
import einops

import cv2
import numpy as np
from PIL import Image
from my_affectgpt.common.registry import registry
from my_affectgpt.models.ImageBind.models.imagebind_model import ImageBindModel, ModalityType
from my_affectgpt.models.ImageBind.models import imagebind_model

# frames: [(b t) h w c]
def func_VideoReader_to_Image(frames):
    outputs = []
    for frame in frames:
        pil_image = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
        outputs.append(pil_image)
    return outputs


## this is the default visual encoder in Video-Llama [takes transformer output as input]
@registry.register_visual_encoder("EVA_CLIP_G")
class EVA_CLIP_G(Blip2Base):
    def __init__(self):
        super(EVA_CLIP_G, self).__init__()

        # use default parameters
        self.use_grad_checkpoint = False
        self.vit_precision = "fp16"
        self.img_size = 224
        self.drop_path_rate = 0
        self.num_query_token = 32


        print('====== Loading VIT ======')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(config.PATH_TO_VISUAL["EVA_CLIP_G"], 
                                                                       self.img_size, self.drop_path_rate, 
                                                                       self.use_grad_checkpoint, self.vit_precision)
        
        ## freeze the weights
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()

        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()

        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        self.visual_encoder.train = disabled_train
        self.ln_vision.train = disabled_train


        print('====== Loading VIT Q-Former ======')
        # the encoder's output used to be [32, 1408]
        self.Qformer, self.query_tokens = self.init_Qformer(self.num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=config.PATH_TO_VISUAL['VIT_QFORMER'])
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train
        self.query_tokens.requires_grad = False

        self.hidden_size = self.Qformer.config.hidden_size # final output feature dim [32, 768]
        print('====== All these parameters are fixed during training!! ======')


    # image encoding: [b c t h w] => [b t q h]
    def forward(self, image, raw_image):

        device = image.device
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')

        # Image Encoder:
        image_embeds = self.ln_vision(self.visual_encoder(image)) # (b, t, c, h, w) -> (b, t, block=257, 1408)
        
        # + Q-Former => compress each image from (block=257, 1408) to (32, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [(b t), block]
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [1, 32, 768] -> [(b t), 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ) # query_output.last_hidden_state is in [(b t), 32, 768], each image's info compressed to [32, 768]
        q_hidden_state = query_output.last_hidden_state

        # final output format becomes (b t q h)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
        return frame_hidden_state


## version with ViT's Q-Former removed [takes transformer output as input]
@registry.register_visual_encoder("EVA_CLIP_G_NO_QFORMER")
class EVA_CLIP_G_NO_QFORMER(Blip2Base):
    def __init__(self):
        super(EVA_CLIP_G_NO_QFORMER, self).__init__()

        # use default parameters
        self.use_grad_checkpoint = False
        self.vit_precision = "fp16"
        self.img_size = 224
        self.drop_path_rate = 0
        self.num_query_token = 32


        print('====== Loading VIT ======')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(config.PATH_TO_VISUAL["EVA_CLIP_G"], 
                                                                       self.img_size, self.drop_path_rate, 
                                                                       self.use_grad_checkpoint, self.vit_precision)
        
        ## freeze the weights
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()

        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()

        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        self.visual_encoder.train = disabled_train
        self.ln_vision.train = disabled_train

        self.hidden_size = self.visual_encoder.num_features # 1408
        print('====== All these parameters are fixed during training!! ======')


    # image encoding: [b c t h w] => [b t h]
    def forward(self, image, raw_image):

        device = image.device
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')

        # Image Encoder:
        image_embeds = self.ln_vision(self.visual_encoder(image)) # [(b t), c, h, w] -> [(b t), block=257, 1408]
        
        # Compresss
        image_embeds = torch.mean(image_embeds, axis=1) # [(b t) h]
        image_embeds = einops.rearrange(image_embeds, '(b t) h -> b t h', b=batch_size, t=time_length)

        return image_embeds 


## use the functions inside CLIP_VIT_LARGE for feature extraction [takes raw data as input]
@registry.register_visual_encoder("CLIP_VIT_LARGE")
class CLIP_VIT_LARGE(Blip2Base):
    def __init__(self):
        super(CLIP_VIT_LARGE, self).__init__()

        print('====== Loading CLIP_VIT_LARGE ======')
        model_dir = config.PATH_TO_VISUAL['CLIP_VIT_LARGE']
        self.model = AutoModel.from_pretrained(model_dir)
        self.processor  = AutoFeatureExtractor.from_pretrained(model_dir)
        
        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model = self.model.eval()

        self.hidden_size = self.model.config.projection_dim # 768
        print('====== All these parameters are fixed during training!! ======')

    
    # image encoding: [b c t h w] => [b t h]
    def forward(self, image, raw_image):
        device = raw_image.device
        batch_size, _, time_length, _, _ = raw_image.size()
        raw_image = einops.rearrange(raw_image, 'b c t h w -> (b t) h w c')

        raw_image = func_VideoReader_to_Image(raw_image)
        inputs = self.processor(images=raw_image, return_tensors="pt")['pixel_values']
        inputs = inputs.to(device)
        embeddings = self.model.get_image_features(inputs) # [(b, t) h]

        embeddings = einops.rearrange(embeddings, '(b t) h -> b t h', b=batch_size, t=time_length)
        return embeddings


## use DINO2_LARGE for feature extraction [takes raw data as input]
@registry.register_visual_encoder("DINO2_LARGE")
class DINO2_LARGE(Blip2Base):
    def __init__(self):
        super(DINO2_LARGE, self).__init__()

        print('====== Loading DINO2_LARGE ======')
        model_dir = config.PATH_TO_VISUAL['DINO2_LARGE']

        self.model = AutoModel.from_pretrained(model_dir)
        self.processor = AutoImageProcessor.from_pretrained(model_dir)

        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model = self.model.eval()

        self.hidden_size = self.model.config.hidden_size # 1024
        print('====== All these parameters are fixed during training!! ======')


    # image encoding: [b c t h w] => [b t h]
    def forward(self, image, raw_image):
        device = raw_image.device
        batch_size, _, time_length, _, _ = raw_image.size()
        raw_image = einops.rearrange(raw_image, 'b c t h w -> (b t) h w c')

        raw_image = func_VideoReader_to_Image(raw_image)
        inputs = self.processor(images=raw_image, return_tensors="pt")['pixel_values']
        inputs = inputs.to(device)
        embeddings = self.model(inputs, output_hidden_states=True).hidden_states # [(b, t)] * [58, 196 patch + 1 cls, feat=768]
        embeddings = torch.stack(embeddings)[-1].mean(dim=1) # take the last layer's features, averaged over all blocks, [(b t), feat=768]

        embeddings = einops.rearrange(embeddings, '(b t) h -> b t h', b=batch_size, t=time_length)
        return embeddings
    

## use SigLIP_SO for feature extraction [takes raw data as input]
@registry.register_visual_encoder("SigLIP_SO")
class SigLIP_SO(Blip2Base):
    def __init__(self):
        super(SigLIP_SO, self).__init__()

        print('====== Loading SigLIP_SO ======')
        model_dir = config.PATH_TO_VISUAL['SigLIP_SO']
        self.model = AutoModel.from_pretrained(model_dir)
        self.processor  = AutoImageProcessor.from_pretrained(model_dir)
        
        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model = self.model.eval()

        self.hidden_size = self.model.config.vision_config.hidden_size # 1152
        print('====== All these parameters are fixed during training!! ======')


    # image encoding: [b c t h w] => [b t h]
    def forward(self, image, raw_image):
        device = raw_image.device
        batch_size, _, time_length, _, _ = raw_image.size()
        raw_image = einops.rearrange(raw_image, 'b c t h w -> (b t) h w c')

        raw_image = func_VideoReader_to_Image(raw_image)
        inputs = self.processor(images=raw_image, return_tensors="pt")['pixel_values']
        inputs = inputs.to(device)
        embeddings = self.model.vision_model(inputs, output_hidden_states=True).hidden_states # [(b, t)] * [58, 196 patch + 1 cls, feat=768]
        embeddings = torch.stack(embeddings)[-1].mean(dim=1) # take the last layer's features, averaged over all blocks, [(b t), feat=768]

        embeddings = einops.rearrange(embeddings, '(b t) h -> b t h', b=batch_size, t=time_length)
        return embeddings


## register the ImageBind acoustic encoder [takes transformer output as input]
@registry.register_acoustic_encoder("IMAGEBIND")
class IMAGEBIND(Blip2Base):
    def __init__(self):
        super(IMAGEBIND, self).__init__()

        print('====== Loading IMAGEBIND ======')
        model_dir = config.PATH_TO_AUDIO['IMAGEBIND']

        self.audio_encoder, self.hidden_size = imagebind_model.imagebind_huge()
        self.audio_encoder.load_state_dict(torch.load(model_dir, weights_only=True))

        ## freeze the weights
        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder.eval()

        print('====== All these parameters are fixed during training!! ======')


    # audio:     [b, t, 1, 128, 204] # stores mel spec
    # raw_audio: [b, t, 1, 32000] # stores sample points
    def forward(self, audio, raw_audio):
        device = audio.device
        _, embeddings = self.audio_encoder.get_audio_feature(audio, modality_type=ModalityType.AUDIO)
        return embeddings


## register the DATA2VEC_BASE acoustic encoder [takes raw data as input]
@registry.register_acoustic_encoder("DATA2VEC_BASE")
class DATA2VEC_BASE(Blip2Base):
    def __init__(self):
        super(DATA2VEC_BASE, self).__init__()

        print('====== Loading DATA2VEC_BASE ======')
        model_dir = config.PATH_TO_AUDIO['DATA2VEC_BASE']

        self.model = AutoModel.from_pretrained(model_dir)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

        print('====== All these parameters are fixed during training!! ======')


    # audio:     [b, t, 1, 128, 204] # stores mel spec
    # raw_audio: [b, t, 1, 32000] # stores sample points
    def forward(self, audio, raw_audio):
        device = raw_audio.device
        raw_audio = raw_audio[:,:,0,:] # [b, t, s]
        batch_size, time_length, _ = raw_audio.size()
        raw_audio = einops.rearrange(raw_audio, 'b t s -> (b t) s')

        layer_ids = [-4, -3, -2, -1]
        input_values = self.feature_extractor(raw_audio, sampling_rate=16000, return_tensors="pt").input_values # [(b t), s]
        input_values = input_values.to(device)
        hidden_states = self.model(input_values[0], output_hidden_states=True).hidden_states # tuple of ((b t), T, D)
        feature = torch.stack(hidden_states)[layer_ids].mean(dim=0)  # ((b t), T, D)
        feature = feature.mean(dim=1) # ((b t), D)
        embeddings = einops.rearrange(feature, '(b t) h -> b t h', b=batch_size, t=time_length)

        return embeddings


## register the WAVLM_LARGE acoustic encoder [takes raw data as input]
@registry.register_acoustic_encoder("WAVLM_LARGE")
class WAVLM_LARGE(Blip2Base):
    def __init__(self):
        super(WAVLM_LARGE, self).__init__()

        print('====== Loading WAVLM_LARGE ======')
        model_dir = config.PATH_TO_AUDIO['WAVLM_LARGE']

        self.model = AutoModel.from_pretrained(model_dir)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

        print('====== All these parameters are fixed during training!! ======')


    # audio:     [b, t, 1, 128, 204] # stores mel spec
    # raw_audio: [b, t, 1, 32000] # stores sample points
    def forward(self, audio, raw_audio):
        device = raw_audio.device
        raw_audio = raw_audio[:,:,0,:] # [b, t, s]
        batch_size, time_length, _ = raw_audio.size()
        raw_audio = einops.rearrange(raw_audio, 'b t s -> (b t) s')

        layer_ids = [-4, -3, -2, -1]
        input_values = self.feature_extractor(raw_audio, sampling_rate=16000, return_tensors="pt").input_values # [(b t), s]
        input_values = input_values.to(device)
        hidden_states = self.model(input_values[0], output_hidden_states=True).hidden_states # tuple of ((b t), T, D)
        feature = torch.stack(hidden_states)[layer_ids].mean(dim=0)  # ((b t), T, D)
        feature = feature.mean(dim=1) # ((b t), D)
        embeddings = einops.rearrange(feature, '(b t) h -> b t h', b=batch_size, t=time_length)

        return embeddings


@registry.register_acoustic_encoder("HUBERT_LARGE")
class HUBERT_LARGE(Blip2Base):
    def __init__(self):
        super(HUBERT_LARGE, self).__init__()

        print('====== Loading HUBERT_LARGE ======')
        model_dir = config.PATH_TO_AUDIO['HUBERT_LARGE']

        self.model = AutoModel.from_pretrained(model_dir)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

        ## freeze the weights
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

        print('====== All these parameters are fixed during training!! ======')


    # audio:     [b, t, 1, 128, 204] # stores mel spec
    # raw_audio: [b, t, 1, 32000] # stores sample points
    def forward(self, audio, raw_audio):
        device = raw_audio.device
        raw_audio = raw_audio[:,:,0,:] # [b, t, s]
        batch_size, time_length, _ = raw_audio.size()
        raw_audio = einops.rearrange(raw_audio, 'b t s -> (b t) s')

        layer_ids = [-4, -3, -2, -1]
        input_values = self.feature_extractor(raw_audio, sampling_rate=16000, return_tensors="pt").input_values # [(b t), s]
        input_values = input_values.to(device)
        hidden_states = self.model(input_values[0], output_hidden_states=True).hidden_states # tuple of ((b t), T, D)
        feature = torch.stack(hidden_states)[layer_ids].mean(dim=0)  # ((b t), T, D)
        feature = feature.mean(dim=1) # ((b t), D)
        embeddings = einops.rearrange(feature, '(b t) h -> b t h', b=batch_size, t=time_length)

        return embeddings


