import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets

# for crossdis, we only evaluate four class => [happy, sad, neutral, angry]
emo2idx = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3}
dataset_map = {
    'IEMOCAPFour':  {0: 'happy', 1: 'sad',   2: 'neutral', 3: 'angry'},
    'IEMOCAPSix':   {0: 'happy', 1: 'sad',   2: 'neutral', 3: 'angry'},
    'MELD':         {0: 'angry', 1: 'happy', 2: 'sad',     3: 'neutral'},
    'MER2023':      {'neutral': 'neutral', 'angry': 'angry', 'happy': 'happy', 'sad': 'sad'}
    }


class CROSSDIS:
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        
        args.output_dim1 = 4
        args.output_dim2 = 0
        args.metric_name = 'emo'

    def get_loaders(self):
        
        train_loaders, eval_loaders, test_loaders = [], [], []

        ################ read train loader ################
        self.args.dataset = self.train_dataset
        self.args.snr = self.args.train_snr
        if self.train_dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            names, labels = self.read_names_labels(self.train_dataset, data_type='whole', debug=self.debug)
            train_eval_idxs = self.gain_cv_for_iemocap(names)
            whole_dataset = get_datasets(self.args, names, labels)
            ## gain 5-fold results
            for ii in range(len(train_eval_idxs)):
                train_idxs = train_eval_idxs[ii][0]
                eval_idxs  = train_eval_idxs[ii][1]
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
                train_loaders.append(train_loader)
                eval_loaders.append(eval_loader)
        elif self.train_dataset in ['MER2023']:
            names, labels = self.read_names_labels(self.train_dataset, data_type='train', debug=self.debug)
            train_dataset = get_datasets(self.args, names, labels)
            # gain indices for cross-validation
            train_eval_idxs = self.gain_cv_for_mer2023(len(names), 5)
            ## gain train and eval loaders
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
        elif self.train_dataset in ['MELD']:
            names, labels = self.read_names_labels(self.train_dataset, data_type='train', debug=self.debug)
            dataset = get_datasets(self.args, names, labels)
            train_loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    collate_fn=dataset.collater,
                                    pin_memory=True)
            train_loaders.append(train_loader)
            names, labels = self.read_names_labels(self.train_dataset, data_type='val', debug=self.debug)
            dataset = get_datasets(self.args, names, labels)
            eval_loader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        shuffle=False,
                                        pin_memory=True)
            eval_loaders.append(eval_loader)


        ################ read test loader ################
        self.args.dataset = self.test_dataset
        self.args.snr = self.args.test_snr
        if self.test_dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            data_type = 'whole'
        elif self.test_dataset in ['MER2023']:
            data_type = 'test1'
        elif self.test_dataset in ['MELD']:
            data_type = 'test'
        names, labels = self.read_names_labels(self.test_dataset, data_type, debug=self.debug)
        test_dataset = get_datasets(self.args, names, labels)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 collate_fn=test_dataset.collater,
                                 shuffle=False,
                                 pin_memory=True)
        test_loaders.append(test_loader)


        return train_loaders, eval_loaders, test_loaders
    

    def func_uniform_labels(self, corpus, mapping):
        names, labels = [], []
        for name in corpus:
            label = corpus[name]
            if label['emo'] in mapping:
                label['emo'] = emo2idx[mapping[label['emo']]]
                names.append(name)
                labels.append(label)
        return names, labels
    
    def read_names_labels(self, dataset, data_type, debug=False):
        # gain label_path
        label_path = config.PATH_TO_LABEL[dataset]
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'val':   corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        if data_type == 'test':  corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        if data_type == 'whole': corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist() # for IEMOCAP
        if data_type == 'test1': corpus = np.load(label_path, allow_pickle=True)['test1_corpus'].tolist() # for MER2023
        # label mapping to same four class discrete labels [remove not included samples]
        print (f'pre sample number: {len(corpus)}')
        names, labels = self.func_uniform_labels(corpus, dataset_map[dataset])
        print (f'after sample number: {len(names)}')
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels

    ## gain cv index for iemocap => leave-one-session-out
    def gain_cv_for_iemocap(self, names):
        session_to_idx = {}
        for idx, vid in enumerate(names):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        train_test_idxs = []
        for ii in range(5): # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(5):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])
        return train_test_idxs
    
    ## gain cv index for mer2023 => random split
    def gain_cv_for_mer2023(self, whole_num, num_folder):
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

    # 计算标准的ACC, WF1的
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        emo_preds = np.argmax(emo_probs, 1)
        emo_accuracy = accuracy_score(emo_labels, emo_preds)
        emo_fscore = f1_score(emo_labels, emo_preds, average='weighted')

        results = {
                    'emoprobs':  emo_probs,
                    'emolabels': emo_labels,
                    'emoacc':    emo_accuracy,
                    'emofscore': emo_fscore
                    }
        outputs = f'f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}'

        return results, outputs
