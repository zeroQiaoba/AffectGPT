import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets


# MER 测试的时候，是将 train 随机分成5份进行cv，test包括 [test1, test2, test3]
class IEMOCAP:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.num_folder = 5
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]

        self.dataset = args.dataset
        assert self.dataset in ['IEMOCAPFour', 'IEMOCAPSix']

        # update args
        if self.dataset == 'IEMOCAPFour':
            args.output_dim1 = 4
            args.output_dim2 = 0
            args.metric_name = 'emo'
        elif self.dataset == 'IEMOCAPSix':
            args.output_dim1 = 6
            args.output_dim2 = 0
            args.metric_name = 'emo'

    def get_loaders(self):
        
        ################ train & eval (five-fold) ################
        self.names, self.labels = self.read_names_labels(self.label_path, debug=self.debug)
        train_eval_idxs = self.split_indexes_using_session()
        print (f'sample number {len(self.names)}')
        self.args.snr = self.args.train_snr
        train_dataset = get_datasets(self.args, self.names, self.labels)
        self.args.snr = self.args.test_snr
        eval_dataset = get_datasets(self.args, self.names, self.labels)

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
            eval_loader = DataLoader(eval_dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(eval_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=eval_dataset.collater,
                                     pin_memory=True)
            train_loaders.append(train_loader)
            eval_loaders.append(eval_loader)

        return train_loaders, eval_loaders, []
    

    def read_names_labels(self, label_path, debug=False):
        names, labels = [], []
        corpus = np.load(label_path, allow_pickle=True)['whole_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels


    ## 生成 n-folder 交叉验证需要的index信息
    def split_indexes_using_session(self):
        
        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(self.names):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == self.num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(self.num_folder): # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(self.num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        return train_test_idxs

    # IEMOCAP是计算标准的ACC, WF1的
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
    