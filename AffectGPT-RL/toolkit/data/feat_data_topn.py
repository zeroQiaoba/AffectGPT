import torch
import numpy as np
from torch.utils.data import Dataset

from toolkit.globals import *
from toolkit.utils.read_data import *

## select top performed features for each modality
class Data_Feat_TOPN(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        assert args.fusion_topn is not None and args.fusion_modality in ['AVT','AV', 'AT', 'VT']
        if args.fusion_modality == 'AVT':
            RANK_LIST = [AUDIO_RANK_LOW2HIGH, TEXT_RANK_LOW2HIGH,  IMAGR_RANK_LOW2HIGH]
        elif args.fusion_modality == 'AT':
            RANK_LIST = [AUDIO_RANK_LOW2HIGH, TEXT_RANK_LOW2HIGH,  TEXT_RANK_LOW2HIGH]
        elif args.fusion_modality == 'AV':
            RANK_LIST = [AUDIO_RANK_LOW2HIGH, IMAGR_RANK_LOW2HIGH, IMAGR_RANK_LOW2HIGH]
        elif args.fusion_modality == 'VT':
            RANK_LIST = [TEXT_RANK_LOW2HIGH,  TEXT_RANK_LOW2HIGH,  IMAGR_RANK_LOW2HIGH]

        # gain all feature names
        featnames = []
        for RANK in RANK_LIST:
            featnames.extend(RANK[-args.fusion_topn:])
        assert len(featnames) == args.fusion_topn * 3
        featnames = [func_name_conversion(feature, suffix='UTT') for feature in featnames]
        featroots = [os.path.join(feat_root, feature) for feature in featnames]
        print (f'feature number: {len(featroots)}')

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale == 1
        assert self.feat_type == 'utt'

        # read datas (reduce __getitem__ durations)
        whole_features, whole_dims = [], []
        for featroot in featroots:
            features, featdim = func_read_multiprocess(featroot, self.names, read_type='feat')
            for ii in range(len(features)):
                features[ii] = np.mean(features[ii], axis=0)
            whole_features.append(features)
            whole_dims.append(featdim)
        self.whole_features = whole_features
        self.whole_dims = whole_dims

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
        instance = dict(
            emo   = self.labels[index]['emo'],
            val   = self.labels[index]['val'],
            name  = self.names[index],
        )
        for ii, features in enumerate(self.whole_features):
            instance[f'feat{ii}'] = features[index]
        return instance
    

    def collater(self, instances):

        batch = dict()
        for ii in range(len(self.whole_features)):
            batch[f'feat{ii}'] = [instance[f'feat{ii}'] for instance in instances]
            batch[f'feat{ii}'] = torch.FloatTensor(np.array(batch[f'feat{ii}']))

        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'topn feature dims: {self.whole_dims}')
        return self.whole_dims, self.whole_dims, self.whole_dims
    