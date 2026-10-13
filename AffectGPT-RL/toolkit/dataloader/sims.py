import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets

class SIMS:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]

        self.dataset = args.dataset
        assert self.dataset in ['SIMS']
        
        # update args
        args.output_dim1 = 0
        args.output_dim2 = 1
        args.metric_name = 'emo'

    def get_loaders(self):
        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            names, labels = self.read_names_labels(self.label_path, data_type, debug=self.debug)
            print (f'{data_type}: sample number {len(names)}')
           
            if data_type in ['train', 'val']:
                self.args.snr = self.args.train_snr
                dataset = get_datasets(self.args, names, labels)
            else:
                self.args.snr = self.args.test_snr
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
    

    def read_names_labels(self, label_path, data_type, debug=False):
        names, labels = [], []
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'val':   corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        if data_type == 'test':  corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels


    # 与CMU数据集一致，先采用acc-2作为结果指标
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
    