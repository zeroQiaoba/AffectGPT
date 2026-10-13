import numpy as np

import torch
import random
from torch.utils.data import Dataset

from toolkit.utils.functions import *
from toolkit.utils.e2e_utils import *
from toolkit.utils.read_data import *

class Data_E2E(Dataset):
    def __init__(self, args, names, labels):

        self.names = names
        self.labels = labels

        # gain path
        self.label_path = config.PATH_TO_LABEL[args.dataset]
        self.audio_root = config.PATH_TO_RAW_AUDIO[args.dataset]
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
        print (f'{self.modality} with pretrained model => {self.model_name}')

        # gain processor
        self.processor = load_e2e_pretrain_processor(self.model_name)

        # pre-load data: (reduce __getitem__ durations) => pre-load may cause OOM [remove this process]
        if self.modality == 'text':
            self.name2trans = func_gain_name2trans(self.trans_path)
        
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

        if self.modality == 'text':
            subtitle = self.name2trans[name]
            if subtitle == '': subtitle = '没有字幕信息。' # => 避免微调模型会出现nan问题
            input_ids = self.processor(subtitle,
                                       return_tensors="pt",
                                       padding="longest",
                                       max_length=1000,
                                       truncation=True,
                                       add_special_tokens=False).input_ids[0]
            instance['input_ids'] = input_ids

        elif self.modality == 'video':
            if self.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE]: # 这个我们之前测试的时候是采用全帧的，那么e2e也得用全帧的 [设置的大一点吧]
                # frames = load_video_from_npy(name, n_frms=16, readtype='uniform', return_raw=True) # [16, 224, 224, 3]
                frames = load_video_from_npy(name, n_frms=60, readtype='uniform', return_raw=True) # [60, 224, 224, 3]
                images = [func_decord_to_image(frame) for frame in frames]
                video = self.processor(images=images)['pixel_values'] # [16, 3, 224, 224]
            elif self.model_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]: # 之前测试的时候，我们也是这么采的，均匀16帧
                frames = load_video_from_npy(name, n_frms=16, readtype='uniform', return_raw=True) # [16, 224, 224, 3]
                video = self.processor(list(frames))['pixel_values'][0] # [16, 3, 224, 224]
            instance['video'] = video

        elif self.modality == 'audio':
            audio_path = func_gain_audiopath(self.audio_root, name)
            waveform = load_and_transform_audio_data([audio_path], "cpu", sample_rate=16000, return_raw=True) # [1, 8, 32000] all audio convert into an unified format
            audio = self.processor(list(waveform[0]), sampling_rate=16000).input_values[0] # [8, 32000]
            instance['audio'] = audio

        return instance
    
    def collater(self, instances):
        
        if self.modality == 'text':
            batch_input_ids = [instance['input_ids'] for instance in instances]
            batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, 
                                                              batch_first=True, 
                                                              padding_value=self.processor.pad_token_id)
            batch = dict(
                input_ids=batch_input_ids, # [batch, seqlen]
                attention_masks=batch_input_ids.ne(self.processor.pad_token_id), # mask padded input
            )
        
        elif self.modality == 'video': # probabliy N * [60, 3, 224, 224]
            videos = np.array([instance['video'] for instance in instances])
            batch = dict(videos = torch.FloatTensor(videos)) # [batch, 8, 16000 * 2]
    
        elif self.modality == 'audio':    
            audios = np.array([instance['audio'] for instance in instances])
            batch = dict(audios = torch.FloatTensor(audios)) # [batch, 8, 16000 * 2]
            
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names


    def get_featdim(self):
        return -1, -1, -1
