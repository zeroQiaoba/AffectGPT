import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from ..utils.e2e_utils import *
from ..globals import *

class E2E_MODEL(nn.Module):
    def __init__(self, args):
        super(E2E_MODEL, self).__init__()

        if args.e2e_name in WHOLE_AUDIO:
            modality = 'audio'
        elif args.e2e_name in WHOLE_TEXT:
            modality = 'text'
        elif args.e2e_name in WHOLE_IMAGE:
            modality = 'video'
        self.modality = modality
        self.model_name = args.e2e_name
        print (f'{modality} with pretrained model => {args.e2e_name}')

        # Params analysis
        feat_dim = args.e2e_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        # define and load pre-trained models
        self.pretrain_model = load_e2e_pretrain_model(self.model_name)
        self.encoder = MLPEncoder(feat_dim, hidden_dim, dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
    
    def forward(self, batch):
        
        if self.modality == 'text':
            input_ids = batch['input_ids']
            attention_masks = batch['attention_masks']
            outputs = self.pretrain_model(input_ids=input_ids,attention_mask=attention_masks,output_hidden_states=True).hidden_states
            outputs = torch.stack(outputs)[[-1, -2, -3, -4]].sum(dim=0) # [batch, seqlen, dim=4096]
            # !! 只计算attention mask=1位置的平均值
            outputs = outputs * torch.unsqueeze(attention_masks, dim=2) # [batch, seqlen, dim] * [batch, seqlen, 1] -> sum 压缩到 [batch, dim]
            features = torch.sum(outputs, dim=1) / torch.sum(attention_masks, dim=1, keepdim=True) # -> [batch, dim]
        
        elif self.modality == 'audio':
            audios = batch['audios'] # [batch, segment=8, 32000]
            bsize, segment, points = audios.size()
            audios = audios.view(-1, points) # [batch*segment, 32000]
            outputs = self.pretrain_model(audios, output_hidden_states=True).hidden_states # tuple * [batch*segment, seqlen, featdim]
            outputs = torch.stack(outputs)[[-1, -2, -3, -4]].sum(dim=0) # [batch*segment, seqlen, featdim]
            features = outputs.mean(dim=1).view(bsize, segment, -1).mean(dim=1) # [batch*segment, featdim] -> [batch, segment, featdim] -> [batch, featdim]

        elif self.modality == 'video':
            videos = batch['videos'] # [bsize, 16, 3, 224, 224]
            if self.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE]:
                bsize, segment, c, h, w = videos.size()
                videos = videos.view(-1, c, h, w) # [bsize * segment, 3, 224, 224]
                outputs = self.pretrain_model.get_image_features(videos) # [bsize * segment, 3, 224, 224] -> [bsize * segment, featdim]
                features = outputs.view(bsize, segment, -1).mean(dim=1) # [bsize, featdim]
            elif self.model_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]:
                bsize = videos.size(0) # [bsize=32, segment=16, 3, 224, 224]
                outputs = self.pretrain_model(videos).last_hidden_state # [bsize, 1586, 768]
                num_patches_per_frame = (self.pretrain_model.config.image_size // self.pretrain_model.config.patch_size) ** 2 # 14*14
                outputs = outputs.view(bsize, 16 // self.pretrain_model.config.tubelet_size, num_patches_per_frame, -1) # [bsize, segment=8, patch, featdim]
                features = outputs.mean(dim=2).mean(dim=1) # [bsize, featdim]

        features = self.encoder(features)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
