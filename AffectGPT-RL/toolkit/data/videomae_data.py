import numpy as np

import torch
import random
from torch.utils.data import Dataset

from toolkit.utils.functions import *
from toolkit.utils.e2e_utils import *
from toolkit.utils.read_data import *


# 如果预先全部加载，会出现内存爆炸的情况
class Data_VIDEOMAE(Dataset):
    def __init__(self, args, names, labels):

        self.names = names
        self.labels = labels

        # gain path
        self.label_path = config.PATH_TO_LABEL[args.dataset]
        self.video_root = config.PATH_TO_RAW_VIDEO[args.dataset]
        self.trans_path = config.PATH_TO_TRANSCRIPTIONS[args.dataset]

        # gain (modality, model_name)
        if args.e2e_name in WHOLE_AUDIO:
            modality = 'audio'
        elif args.e2e_name in WHOLE_TEXT:
            modality = 'text'
        elif args.e2e_name in WHOLE_IMAGE:
            modality = 'video'
        self.modality = modality
        self.model_name = args.e2e_name
        assert args.e2e_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]
        print (f'{self.modality} with pretrained model => {self.model_name}')

        # gain processor
        self.processor = load_e2e_pretrain_processor(self.model_name)

        # debug
        if args.debug:
            instances = []
            for _ in range(32):
                index = random.randint(0, len(self.names)-1)
                instances.append(self.__getitem__(index))
            self.collater(instances)

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        name  = self.names[index]
        instance = dict(
            emo = self.labels[index]['emo'],
            val = self.labels[index]['val'],
            name = name,
        )

        # 相比于preload for accelate, 这种操作start是随机的
        frames = load_video_from_npy(name, n_frms=16, readtype='continuous_polish', return_raw=True) # [16, 224, 224, 3]
        instance['video'] = self.processor(list(frames))['pixel_values'][0] # [16, 3, 224, 224]
        return instance
        
    def collater(self, instances):
        
        videos = np.array([instance['video'] for instance in instances])
        batch = dict(videos = torch.FloatTensor(videos)) # [batch, 8, 3, 224, 224]
    
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names


    def get_featdim(self):
        return -1, -1, -1
