import math
import torch
import torch.nn as nn
import numpy as np
from transformers import VideoMAEForPreTraining

from ..globals import *
from ..utils.e2e_utils import *

class VIDEOMAE_PRETRAIN(nn.Module):
    def __init__(self, args):
        super(VIDEOMAE_PRETRAIN, self).__init__()

        if args.e2e_name in WHOLE_AUDIO:
            modality = 'audio'
        elif args.e2e_name in WHOLE_TEXT:
            modality = 'text'
        elif args.e2e_name in WHOLE_IMAGE:
            modality = 'video'
        self.modality = modality
        self.e2e_name = args.e2e_name
        assert self.e2e_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]
        print (f'{modality} with pretrained model => {args.e2e_name}')

        # Params analysis
        self.grad_clip = args.grad_clip
        self.mae_mask_ratio = args.mae_mask_ratio

        # define and load pre-trained models
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{self.e2e_name}')
        self.model = VideoMAEForPreTraining.from_pretrained(model_dir)
        # model = AutoModel.from_pretrained(model_dir)
        # model = VideoMAEModel.from_pretrained(model_dir)
        # model = VideoMAEForPreTraining.from_pretrained(model_dir)
        # model.save_pretrained('./temp')

    def forward(self, batch):
        
        videos = batch['videos']
        bsize = videos.size(0) # [bs=32, 16, 3, 224, 224]
        num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
        seq_length = (16 // self.model.config.tubelet_size) * num_patches_per_frame
        
        # gain mask: [ensure the number of masked items is identify for each sample]
        batch_bool_masked_pos = []
        mask_num = math.ceil(seq_length * self.mae_mask_ratio)
        for _ in range(bsize):
            bool_masked_pos = np.ones(seq_length)
            mask = np.random.choice(seq_length, mask_num, replace=False)
            bool_masked_pos[mask] = 0
            bool_masked_pos = torch.as_tensor(bool_masked_pos).bool().unsqueeze(0)
            batch_bool_masked_pos.append(bool_masked_pos)
        batch_bool_masked_pos = torch.cat(batch_bool_masked_pos) # [32, 1568]
        batch_bool_masked_pos = batch_bool_masked_pos.cuda()

        outputs = self.model(videos, bool_masked_pos=batch_bool_masked_pos)
        interloss = outputs.loss

        return [], [], [], interloss
