import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets

class MELD:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]

        self.dataset = args.dataset
        assert self.dataset in ['MELD']
        
        # update args
        args.output_dim1 = 7
        args.output_dim2 = 0
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


    # MELD 测试 7-emo classification performance
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
    
   