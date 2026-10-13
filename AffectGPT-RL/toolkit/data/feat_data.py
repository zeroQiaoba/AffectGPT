import torch
import numpy as np
from torch.utils.data import Dataset
from toolkit.utils.read_data import *

class Data_Feat(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        if args.snr is None: # 通过snr控制特征读取位置
            audio_root = os.path.join(feat_root, args.audio_feature)
            text_root  = os.path.join(feat_root, args.text_feature )
            video_root = os.path.join(feat_root, args.video_feature)
        else:
            # data2vec-audio-base-960h-UTT -> data2vec-audio-base-960h-noisesnrmix-UTT
            # eGeMAPS_UTT -> eGeMAPS_noisesnrmix_UTT
            audio_root = os.path.join(feat_root, args.audio_feature[-4].join([args.audio_feature[:-4], args.snr, 'UTT']))
            text_root  = os.path.join(feat_root, args.text_feature[-4].join([args.text_feature[:-4], args.snr, 'UTT']))
            video_root = os.path.join(feat_root, args.video_feature[-4].join([args.video_feature[:-4], args.snr, 'UTT']))
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        # step2: align to batch
        if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
            audios, texts, videos = align_to_utt(audios, texts, videos)
        elif self.feat_type == 'frm_align':
            audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        elif self.feat_type == 'frm_unalign':
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = dict(
            audio = self.audios[index],
            text  = self.texts[index],
            video = self.videos[index],
            emo   = self.labels[index]['emo'],
            val   = self.labels[index]['val'],
            name  = self.names[index],
        )
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]

        batch = dict(
            audios = torch.FloatTensor(np.array(audios)),
            texts  = torch.FloatTensor(np.array(texts)),
            videos = torch.FloatTensor(np.array(videos)),
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
    