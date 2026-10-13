"""
get_models: get models and load default configs; 
link: https://github.com/thuiar/MMSA-FET/tree/master
"""
import torch.nn as nn

from .tfn import TFN
from .lmf import LMF
from .mfn import MFN
from .mfm import MFM
from .mult import MULT
from .misa import MISA
from .mctn import MCTN
from .mmim import MMIM
from .lf_dnn import LF_DNN
from .ef_lstm import EF_LSTM
from .graph_mfn import Graph_MFN
from .attention import Attention
from .attention_topn import Attention_TOPN

from .e2e_model import E2E_MODEL

from .videomae_pretrain import VIDEOMAE_PRETRAIN

class get_models(nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            'attention': Attention,
            'lf_dnn': LF_DNN,
            'lmf': LMF,
            'misa': MISA,
            'mmim': MMIM,
            'tfn': TFN,
            
            # 只支持align
            'mfn': MFN, # slow
            'graph_mfn': Graph_MFN, # slow
            'ef_lstm': EF_LSTM, 
            'mfm': MFM, # slow
            'mctn': MCTN, # slow

            # 支持align/unalign
            'mult': MULT, # slow

            # e2e model 支持raw
            'e2e_model': E2E_MODEL,

            # 模型预训练
            'videomae_pretrain': VIDEOMAE_PRETRAIN,

            # 支持每个模态选择topn特征输入
            'attention_topn': Attention_TOPN,

        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
