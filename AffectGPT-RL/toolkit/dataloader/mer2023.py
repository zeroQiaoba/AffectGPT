import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets

# MER 测试的时候，是将 train 随机分成5份进行cv，test包括 [test1, test2, test3]
class MER2023:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.num_folder = 5
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]

        self.dataset = args.dataset
        assert self.dataset in ['MER2023']

        # update args
        args.output_dim1 = 6
        args.output_dim2 = 1
        args.metric_name = 'emoval'
      

    def get_loaders(self):
        
        ################ train & eval (five-fold) ################        
        data_type = 'train'
        names, labels = self.read_names_labels(self.label_path, data_type, debug=self.debug)
        print (f'{data_type}: sample number {len(names)}')
        self.args.snr = self.args.train_snr
        train_dataset = get_datasets(self.args, names, labels)

        # gain indices for cross-validation
        whole_num = len(names)
        train_eval_idxs = self.random_split_indexes(whole_num, self.num_folder)

        ## gain train and eval loaders
        train_loaders = []
        eval_loaders = []
        for ii in range(len(train_eval_idxs)):
            train_idxs = train_eval_idxs[ii][0]
            eval_idxs  = train_eval_idxs[ii][1]
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      num_workers=self.num_workers,
                                      collate_fn=train_dataset.collater,
                                      pin_memory=True)
            eval_loader = DataLoader(train_dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(eval_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=train_dataset.collater,
                                     pin_memory=True)
            train_loaders.append(train_loader)
            eval_loaders.append(eval_loader)

        ################ test sets (test1 & test2 & test3) ################
        test_loaders = []
        for data_type in ['test1', 'test2', 'test3']:
            names, labels = self.read_names_labels(self.label_path, data_type, debug=self.debug)
            print (f'{data_type}: sample number {len(names)}')
            self.args.snr = self.args.test_snr
            test_dataset = get_datasets(self.args, names, labels)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     collate_fn=test_dataset.collater,
                                     shuffle=False,
                                     pin_memory=True)
            test_loaders.append(test_loader)
        
        return train_loaders, eval_loaders, test_loaders
    
    
    # read (names, labels)
    def read_names_labels(self, label_path, data_type, debug=False):
        names, labels = [], []
        assert data_type in ['train', 'test1', 'test2', 'test3']
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist()
        if data_type == 'test2': corpus = np.load(label_path, allow_pickle=True)['test2_corpus'].tolist()
        if data_type == 'test3': corpus = np.load(label_path, allow_pickle=True)['test3_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # post process for labels
        for ii, label in enumerate(labels):
            emo = emo2idx_mer[label['emo']]
            if 'val' not in label or label['val'] == '':
                val = -10
            else:
                val = label['val']
            labels[ii] = {'emo': emo, 'val': val}
        # for debug
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
    
    # 对于 MER 数据集，同时计算emo|val|overall三个指标 [不同test不一样]
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        emo_preds = np.argmax(emo_probs, 1)
        emo_accuracy = accuracy_score(emo_labels, emo_preds)
        emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

        val_mse = mean_squared_error(val_labels, val_preds)

        results = {
                    'emoprobs':  emo_probs,
                    'emolabels': emo_labels,
                    'emoacc':    emo_accuracy,
                    'emofscore': emo_fscore,
                    'valpreds':  val_preds,
                    'vallabels': val_labels,
                    'valmse':    val_mse,
                    }
        outputs = f'f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}_val:{val_mse:.4f}'
        return results, outputs

