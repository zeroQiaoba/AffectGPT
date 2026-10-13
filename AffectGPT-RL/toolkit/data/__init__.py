from torch.utils.data import Dataset

from .feat_data import Data_Feat
from .feat_data_topn import Data_Feat_TOPN
from .e2e_data import Data_E2E
from .videomae_data import Data_VIDEOMAE

# 目标：输入 (names, labels, data_type)，得到所有特征与标签
class get_datasets(Dataset):

    def __init__(self, args, names, labels):

        MODEL_DATASET_MAP = {
            
            # 解析特征
            'attention': Data_Feat,
            'lf_dnn': Data_Feat,
            'lmf': Data_Feat,
            'misa': Data_Feat,
            'mmim': Data_Feat,
            'tfn': Data_Feat,
            'mfn': Data_Feat,
            'graph_mfn': Data_Feat,
            'ef_lstm': Data_Feat, 
            'mfm': Data_Feat,
            'mctn': Data_Feat,
            'mult': Data_Feat,

            # 解析原始数据
            'e2e_model': Data_E2E,
            
            # 测试预训练
            'videomae_pretrain': Data_VIDEOMAE, 

            # 兼容多特征输入
            'attention_topn': Data_Feat_TOPN,
        }

        self.dataset_class = MODEL_DATASET_MAP[args.model]
        self.dataset = self.dataset_class(args, names, labels)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def collater(self, instances):
        return self.dataset.collater(instances)
         
    def get_featdim(self):
        return self.dataset.get_featdim()