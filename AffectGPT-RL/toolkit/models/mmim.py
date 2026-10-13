"""
From: https://github.com/declare-lab/Multimodal-Infomax
Paper: Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis (EMNLP 2021)
"""

import torch
from torch import nn

from .modules.encoder import MLPEncoder, LSTMEncoder

class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)

        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive,-1))

        # For Gaussian Distribution Estimation
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos':None, 'neg':None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y) 
            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                a = 17.0795
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))

        return lld, sample_dict, H


class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1) # bs
        nce = -(pos - neg).mean()
        return nce


class Fusion(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(Fusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        return y_2


class MMIM(nn.Module):
    def __init__(self, args):
        super().__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout # 0.3 
        hidden_dim = args.hidden_dim # 128
        cpc_layers = args.cpc_layers # 1
        self.alpha = args.alpha # 0.1
        self.beta = args.beta # 0.1
        self.grad_clip = args.grad_clip


        # modality-specific encoder
        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = hidden_dim,
            y_size = hidden_dim,
            mid_activation = "ReLU",
            last_activation = "Tanh"
        )

        self.mi_ta = MMILB(
            x_size = hidden_dim,
            y_size = hidden_dim,
            mid_activation = "ReLU",
            last_activation = "Tanh"
        )
        
        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = hidden_dim, # to be predicted
            y_size = hidden_dim,
            n_layers = cpc_layers,
            activation = "Tanh"
        )
        self.cpc_zv = CPC(
            x_size = hidden_dim,
            y_size = hidden_dim,
            n_layers = cpc_layers,
            activation = "Tanh"
        )
        self.cpc_za = CPC(
            x_size = hidden_dim,
            y_size = hidden_dim,
            n_layers = cpc_layers,
            activation = "Tanh"
        )

        # Trimodal Settings
        self.fusion_prj = Fusion(
            in_size = hidden_dim * 3,
            hidden_size = hidden_dim,
            dropout = dropout
        )

        # output results
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
    

    # support unaligned frame-level features
    def forward(self, batch):
        
        # modality-specific encoder: [seqlen, hiddendim]
        audio_h  =  self.audio_encoder(batch['audios'])
        text_h   =  self.text_encoder(batch['texts'])
        vision_h =  self.video_encoder(batch['videos'])

        # a->t and v->t distance: [seqlen, hiddendim]
        lld_tv = self.mi_tv(x=text_h, y=vision_h)[0]
        lld_ta = self.mi_ta(x=text_h, y=audio_h)[0]
        lld = lld_tv + lld_ta

        # fusion: [seqlen, hiddendim]
        fusion = self.fusion_prj(torch.cat([text_h, audio_h, vision_h], dim=1))

        # nce loss
        nce_t = self.cpc_zt(text_h,   fusion)
        nce_v = self.cpc_zv(vision_h, fusion)
        nce_a = self.cpc_za(audio_h,  fusion)
        nce = nce_t + nce_v + nce_a
        
        emos_out  = self.fc_out_1(fusion)
        vals_out  = self.fc_out_2(fusion)
        interloss = self.alpha * nce - self.beta * lld
        
        return fusion, emos_out, vals_out, interloss
    
