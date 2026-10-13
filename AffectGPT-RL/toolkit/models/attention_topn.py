'''
Description: unimodal encoder + concat + attention fusion
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder

class Attention_TOPN(nn.Module):
    def __init__(self, args):
        super(Attention_TOPN, self).__init__()
        
        feat_dims = args.audio_dim # store topn feat dims
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.feat_dims = feat_dims

        ## --------------------------------------------------- ##
        # self.encoders = []
        # for dim in feat_dims:
        #     self.encoders.append(MLPEncoder(dim, hidden_dim, dropout).cuda()) # list 不能 cuda，还是这么操作吧
         ## --------------------------------------------------- ##
        # debug: list 不能传 cuda 和不能传梯度，是不是一个意思呢？ => yes
        assert len(feat_dims) <= 3 * 6
        if len(feat_dims) >= 1:  self.encoder0  = MLPEncoder(feat_dims[0],  hidden_dim, dropout)
        if len(feat_dims) >= 2:  self.encoder1  = MLPEncoder(feat_dims[1],  hidden_dim, dropout)
        if len(feat_dims) >= 3:  self.encoder2  = MLPEncoder(feat_dims[2],  hidden_dim, dropout)
        if len(feat_dims) >= 4:  self.encoder3  = MLPEncoder(feat_dims[3],  hidden_dim, dropout)
        if len(feat_dims) >= 5:  self.encoder4  = MLPEncoder(feat_dims[4],  hidden_dim, dropout)
        if len(feat_dims) >= 6:  self.encoder5  = MLPEncoder(feat_dims[5],  hidden_dim, dropout)
        if len(feat_dims) >= 7:  self.encoder6  = MLPEncoder(feat_dims[6],  hidden_dim, dropout)
        if len(feat_dims) >= 8:  self.encoder7  = MLPEncoder(feat_dims[7],  hidden_dim, dropout)
        if len(feat_dims) >= 9:  self.encoder8  = MLPEncoder(feat_dims[8],  hidden_dim, dropout)
        if len(feat_dims) >= 10: self.encoder9  = MLPEncoder(feat_dims[9],  hidden_dim, dropout)
        if len(feat_dims) >= 11: self.encoder10 = MLPEncoder(feat_dims[10], hidden_dim, dropout)
        if len(feat_dims) >= 12: self.encoder11 = MLPEncoder(feat_dims[11], hidden_dim, dropout)
        if len(feat_dims) >= 13: self.encoder12 = MLPEncoder(feat_dims[12], hidden_dim, dropout)
        if len(feat_dims) >= 14: self.encoder13 = MLPEncoder(feat_dims[13], hidden_dim, dropout)
        if len(feat_dims) >= 15: self.encoder14 = MLPEncoder(feat_dims[14], hidden_dim, dropout)
        if len(feat_dims) >= 16: self.encoder15 = MLPEncoder(feat_dims[15], hidden_dim, dropout)
        if len(feat_dims) >= 17: self.encoder16 = MLPEncoder(feat_dims[16], hidden_dim, dropout)
        if len(feat_dims) >= 18: self.encoder17 = MLPEncoder(feat_dims[17], hidden_dim, dropout)
        
        self.attention_mlp = MLPEncoder(hidden_dim * len(feat_dims), hidden_dim, dropout)
        self.fc_att   = nn.Linear(hidden_dim, len(feat_dims))
        
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
    
    def forward(self, batch):
        hiddens = []
        ## --------------------------------------------------- ##
        # for ii, encoder in enumerate(self.encoders):
        #     hiddens.append(encoder(batch[f'feat{ii}']))
        ## --------------------------------------------------- ##
        if len(self.feat_dims) >= 1: hiddens.append(self.encoder0(batch[f'feat0']))
        if len(self.feat_dims) >= 2: hiddens.append(self.encoder1(batch[f'feat1']))
        if len(self.feat_dims) >= 3: hiddens.append(self.encoder2(batch[f'feat2']))
        if len(self.feat_dims) >= 4: hiddens.append(self.encoder3(batch[f'feat3']))
        if len(self.feat_dims) >= 5: hiddens.append(self.encoder4(batch[f'feat4']))
        if len(self.feat_dims) >= 6: hiddens.append(self.encoder5(batch[f'feat5']))
        if len(self.feat_dims) >= 7: hiddens.append(self.encoder6(batch[f'feat6']))
        if len(self.feat_dims) >= 8: hiddens.append(self.encoder7(batch[f'feat7']))
        if len(self.feat_dims) >= 9: hiddens.append(self.encoder8(batch[f'feat8']))
        if len(self.feat_dims) >= 10: hiddens.append(self.encoder9(batch[f'feat9']))
        if len(self.feat_dims) >= 11: hiddens.append(self.encoder10(batch[f'feat10']))
        if len(self.feat_dims) >= 12: hiddens.append(self.encoder11(batch[f'feat11']))
        if len(self.feat_dims) >= 13: hiddens.append(self.encoder12(batch[f'feat12']))
        if len(self.feat_dims) >= 14: hiddens.append(self.encoder13(batch[f'feat13']))
        if len(self.feat_dims) >= 15: hiddens.append(self.encoder14(batch[f'feat14']))
        if len(self.feat_dims) >= 16: hiddens.append(self.encoder15(batch[f'feat15']))
        if len(self.feat_dims) >= 17: hiddens.append(self.encoder16(batch[f'feat16']))
        if len(self.feat_dims) >= 18: hiddens.append(self.encoder17(batch[f'feat17']))

        multi_hidden1 = torch.cat(hiddens, dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack(hiddens, dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]

        features  = fused_feat.squeeze(axis=2) # [32, 128] => 解决batch=1报错的问题
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
