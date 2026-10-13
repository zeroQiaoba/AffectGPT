import os
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ..globals import *
from toolkit.data import get_datasets

# 直接读取所有人脸数据，并随机划分 train|test
class MER2023_UNLABEL:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.face_root = config.PATH_TO_RAW_FACE[args.dataset]
        self.video_root = config.PATH_TO_RAW_VIDEO[args.dataset]
        self.name2len_path = os.path.join(config.DATA_DIR[args.dataset], 'unlabel-name2len.npz')

        self.dataset = args.dataset
        assert self.dataset in ['MER2023_UNLABEL']

        # update args
        args.output_dim1 = 0
        args.output_dim2 = 0
        args.metric_name = 'loss'

    def get_loaders(self):
        
        ################ whole faces ################
        names, labels = self.read_names_labels(self.name2len_path, debug=self.debug)
        print (f'sample number {len(names)}') # 73953 samples -> 32983
        whole_dataset = get_datasets(self.args, names, labels)

        # gain indices for cross-validation
        train_eval_idxs = self.random_split_indexes(len(names), 5)

        ## 虽然分割成五分，但是预训练时仅取1份数据
        train_idxs = train_eval_idxs[0][0]
        eval_idxs  = train_eval_idxs[0][1]
        train_loader = DataLoader(whole_dataset,
                                  batch_size=self.batch_size,
                                  sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                  num_workers=self.num_workers,
                                  collate_fn=whole_dataset.collater,
                                  pin_memory=True)
        eval_loader = DataLoader(whole_dataset,
                                 batch_size=self.batch_size,
                                 sampler=SubsetRandomSampler(eval_idxs),
                                 num_workers=self.num_workers,
                                 collate_fn=whole_dataset.collater,
                                 pin_memory=True)

        return [train_loader], [eval_loader], []
    
    def read_names_labels(self, name2len_path, debug=False):
        
        # 至少为 3s [因为考虑处理的时候删除前后1s的数据，降低噪声]
        names = [] 
        name2len = np.load(name2len_path, allow_pickle=True)['name2len'].tolist()
        for name in name2len: 
            if name2len[name] >= 3*25:
                names.append(name)

        # 生成 labels
        labels = len(names) * [{'emo': 0, 'val': -10}]
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels

    ## 生成 n-folder 交叉验证需要的index信息
    def random_split_indexes(self, whole_num, num_folder):

        # gain indices for cross-validation
        indices = np.arange(whole_num)
        random.shuffle(indices)

        # split indices into five-fold
        whole_folder = []
        each_folder_num = int(whole_num / num_folder)
        for ii in range(num_folder-1):
            each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
            whole_folder.append(each_folder)
        each_folder = indices[each_folder_num*(num_folder-1):]
        whole_folder.append(each_folder)
        assert len(whole_folder) == num_folder
        assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

        ## split into train/eval
        train_eval_idxs = []
        for ii in range(num_folder): # ii in [0, 4]
            eval_idxs = whole_folder[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(whole_folder[jj])
            train_eval_idxs.append([train_idxs, eval_idxs])
        
        return train_eval_idxs
    
    # 不在情感维度进行评价，预训练看的是loss下降轨迹
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        results = {}
        outputs = ""
        return results, outputs

