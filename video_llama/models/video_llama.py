import re
import copy
import einops
import random
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import LlamaTokenizer, BertConfig
from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel, ModalityType
from video_llama.models.ImageBind.models import imagebind_model

@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model,
        q_former_model,
        img_size,
        drop_path_rate,
        use_grad_checkpoint,
        vit_precision,
        num_query_token,
        llama_model,
        low_resource, # use 8 bit and put vit in cpu
        device_8bit,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        frozen_llama_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_llama_proj,
        llama_proj_model,
        num_video_query_token,
        num_audio_query_token,
        imagebind_ckpt_path,
    ):
        super().__init__()

        print('====== Loading other necessary components ======')
        self.low_resource = low_resource # False

        if (frozen_audio_Qformer and frozen_audio_llama_proj) and (not frozen_video_Qformer or not frozen_llama_proj):
            self.train_flag = 1 # training on video branch
        elif (frozen_video_Qformer and frozen_llama_proj) and (not frozen_audio_Qformer or not frozen_audio_llama_proj):
            self.train_flag = 2 # training on audio branch
        elif (not frozen_video_Qformer or not frozen_llama_proj) and (not frozen_audio_Qformer or not frozen_audio_llama_proj):
            self.train_flag = 3 # training on audio+video branchs
            

        print('====== Loading VIT: eva_vit_g.pth ======')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model, img_size, drop_path_rate, 
                                                                       use_grad_checkpoint, vit_precision)
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = disabled_train
        print("freeze: vision encoder (fixed)")


        print('====== Loading VIT Q-Former: blip2_pretrained_flant5xxl.pth ======')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train
        self.query_tokens.requires_grad = False
        print("freeze: VIT Qformer (fixed)")
        

        print('====== Loading LLAMA: vicuna-7b-v0 ======')
        '''
        llama token ids:
            <unk>: 0
            bos|<s>: 1
            eos|pad|</s>: 2
            <ImageHere>: 32000
            <AudioHere>: 32001
        '''
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('freeze: LLAMA Model (fixed)')


        ## Video Q-Former
        print('====== Loading Video Q-Former: provided by Video-LLama ======')
        self.video_frame_position_embedding = nn.Embedding(32, self.Qformer.config.hidden_size) # [32, 768]
        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers=2)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            print('freeze: video_Qformer')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            print('trainable: video_Qformer')
        

        print(f'====== Loading Video LLAMA proj: {llama_proj_model} ======')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            self.llama_proj.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Video Q-Former LLaMA proj')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Video Q-Former LLaMA proj')


        print(f'====== Loading Audio Encoder: {imagebind_ckpt_path} ======')
        self.audio_encoder,self.audio_hidden_size = imagebind_model.imagebind_huge()
        self.audio_encoder.load_state_dict(torch.load(imagebind_ckpt_path))
        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder.eval()
        print("freeze audio encoder (fixed)")

        
        print('====== Loading Audio Q-Former: provided by Video-LLama ======')
        self.num_audio_query_token = num_audio_query_token
        self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
            vision_width=self.audio_hidden_size, num_hidden_layers=2)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size) # [8, 1024]

        if frozen_audio_Qformer:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = False
            self.audio_query_tokens.requires_grad = False
            for name, param in self.audio_position_embedding.named_parameters():
                param.requires_grad = False
            print('freeze: audio_Qformer')
        else:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = True
            self.audio_query_tokens.requires_grad = True
            for name, param in self.audio_position_embedding.named_parameters():
                param.requires_grad = True
            print('trainable: audio_Qformer')
        

        print('====== Loading audio_llama_proj: provided by Video-LLama ======')
        self.audio_llama_proj = nn.Linear(
            self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if frozen_audio_llama_proj:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Audio Q-Former LLaMA proj')
        else:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Audio Q-Former LLaMA proj')


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    # image: image + vis_process
    # includes: VIT + VIT Q-Former + Video Q-Former + video_llama_proj
    def encode_videoQformer_visual(self, image):
        device = image.device
        
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # VIT + Q-Former + Position Embedding
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # Video Q-Former + llama_proj
            frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state
            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    

    # audio: audio + vis_process
    # include: encoder + audio Q-Former + audio_llama_proj
    def encode_audioQformer(self, audio, modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():

            # modality_type: use imagebind and map audio|video into one embedding space
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio, modality_type=modality_type)

            # encoder: for common space embedding
            batch_size, time_length = audio.size()[:2]
            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            # Audio Q-Former + audio_llama_proj
            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state
            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama
    

    '''
    inference prompt:
    <s>###Human: Close your eyes, open your ears and you imagine only based on the sound that <Audio><AudioHere></Audio>. \
    Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
    The subtitle of this video is <Subtitle>{subtitle}</Subtitle>. \
    Now answer my question based on what you have seen, heard, and subtitles. {user_message} ###Assistant:
    '''
    def forward(self, samples):
        
        image = samples['images']
        # print (f"process dataset {samples['dataset']}") # each iter uses different dataset loadera
        aud_embeds, img_embeds = [], []
        if self.train_flag == 1: # only video branch
            img_embeds, _ = self.encode_videoQformer_visual(image) # [b c t h w]
        elif self.train_flag == 2: # only audio branch
            image = einops.rearrange(image, 'b c t h w -> b t c h w')
            img_embeds, _ = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
        elif self.train_flag == 3: # training on audio+video branchs
            img_embeds, _ = self.encode_videoQformer_visual(image)                              # image: [b c t h w] -> [b, 32, 4096]
            if 'audios' in samples: aud_embeds, _ = self.encode_audioQformer(samples['audios']) # audio: [b t c h w] -> [b, 8,  4096]
            
        # temp_input_ids: <ImageHere> -> [0]
        input_ids = samples['input_ids']
        im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
        au_patch_token_id = self.AUDIO_PATCH_TOKEN_ID
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == im_patch_token_id] = 0
        temp_input_ids[temp_input_ids == au_patch_token_id] = 0
        temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

        # replace <ImageHere> and <AudioHere> with features
        cur_image_idx = 0
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):

            ## replace <ImageHere>
            if (cur_input_ids == im_patch_token_id).sum() != 0:
                cur_image_features = img_embeds[cur_image_idx]
                if (cur_input_ids == im_patch_token_id).sum() != self.num_video_query_token:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+self.num_video_query_token, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], 
                                              cur_image_features, 
                                              cur_input_embeds[mask_index_start+self.num_video_query_token:]), dim=0)

            ## replace <AudioHere>
            if (cur_input_ids == au_patch_token_id).sum() != 0:
                cur_audio_features = aud_embeds[cur_image_idx]
                if (cur_input_ids == au_patch_token_id).sum() != self.num_audio_query_token:
                        raise ValueError("The number of image patch tokens should be the same as the number of audio patches.")
                masked_indices = torch.where(cur_input_ids == au_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+self.num_audio_query_token, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], 
                                              cur_audio_features, 
                                              cur_input_embeds[mask_index_start+self.num_audio_query_token:]), dim=0)
            new_input_embeds.append(cur_input_embeds)
            cur_image_idx+=1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        '''
        inputs_embeds:  [<s>###Human: <Video> <ImageHere>*32 </Video> xxx###Assistant: {target}###Human: xxx###Assistant: {target}###, 2, ...   ]
        attention_mask: [1, 1, ...                                                                                                     1, 0, ...]
        targets:        [-100......,                               ................... {target}###-100,..-100,   .........{target}###, -100, ...]
        '''
        targets = samples['labels']
        attention_mask = samples['attention_masks']
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets)
        loss = outputs.loss
        return {"loss": loss}


    @classmethod
    def from_config(cls, cfg):

        # Image Encoder: VIT + VIT Q-Former
        img_size = cfg.get("image_size", 224)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        num_query_token = cfg.get("num_query_token", 32)
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_precision = cfg.get("vit_precision", "fp16")
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        q_former_model = cfg.get("q_former_model", "models/blip2_pretrained_flant5xxl.pth")

        # Audio Encoder
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", "models/imagebind_huge.pth")

        # Audio/Video Q-Former
        llama_proj_model = cfg.get("llama_proj_model", '')
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)
        frozen_audio_llama_proj = cfg.get("frozen_audio_llama_proj", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        # LLaMA
        device_8bit = cfg.get("device_8bit", 0)
        low_resource = cfg.get("low_resource", False)
        llama_model = cfg.get("llama_model", "models/vicuna-7b-v0")

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            num_query_token=num_query_token,
            llama_model=llama_model,
            low_resource=low_resource,
            device_8bit=device_8bit,
            frozen_llama_proj=frozen_llama_proj,
            frozen_audio_llama_proj=frozen_audio_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            llama_proj_model = llama_proj_model
        )

        # priority: ckpt < ckpt_2 < ckpt_3
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)
            
        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        ckpt_path_3 = cfg.get("ckpt_3", "")
        if ckpt_path_3:
            print("Load third Checkpoint: {}".format(ckpt_path_3))
            ckpt = torch.load(ckpt_path_3, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model

    