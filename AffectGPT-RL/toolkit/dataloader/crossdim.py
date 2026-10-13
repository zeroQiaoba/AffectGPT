import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets

class CROSSDIM:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset

        # update args
        args.output_dim1 = 0
        args.output_dim2 = 1
        args.metric_name = 'emo'

    def get_loaders(self):
        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            if data_type in ['train', 'val']:
                self.args.snr = self.args.train_snr
                self.args.dataset = self.train_dataset
                names, labels = self.read_names_labels(self.train_dataset, data_type, debug=self.debug)
                print (f'{data_type}: sample number {len(names)}')
                dataset = get_datasets(self.args, names, labels)
            else:
                self.args.snr = self.args.test_snr
                self.args.dataset = self.test_dataset
                names, labels = self.read_names_labels(self.test_dataset, data_type, debug=self.debug)
                print (f'{data_type}: sample number {len(names)}')
                dataset = get_datasets(self.args, names, labels)

            if data_type in ['train']:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        pin_memory=True)
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        shuffle=False,
                                        pin_memory=True)
            dataloaders.append(dataloader)
        train_loaders = [dataloaders[0]]
        eval_loaders  = [dataloaders[1]]
        test_loaders  = [dataloaders[2]]
                
        return train_loaders, eval_loaders, test_loaders
    

    def read_names_labels(self, dataset, data_type, debug=False):
        # gain label path
        label_path = config.PATH_TO_LABEL[dataset]
        names, labels = [], []
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'val':   corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        if data_type == 'test':  corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # label mapping to same dimension
        if dataset in ['SIMS', 'SIMSv2']:
            for ii in range(len(labels)):
                labels[ii]['val'] = labels[ii]['val'] * 3
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels


    # CMU 采用的指标，是将val转成2分类计算 ACC, WAF [all labels in range [-3, 3]]
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
        emo_accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
        emo_fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')

        results = { 
                    'valpreds':  val_preds,
                    'vallabels': val_labels,
                    'emoacc':    emo_accuracy,
                    'emofscore': emo_fscore
                    }
        outputs = f'f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}'

        return results, outputs
    