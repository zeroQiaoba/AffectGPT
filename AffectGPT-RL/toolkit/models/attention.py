'''
Description: unimodal encoder + concat + attention fusion
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder, LSTMEncoder

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        
        text_dim    = args.text_dim
        audio_dim   = args.audio_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)

        self.fc_att   = nn.Linear(hidden_dim, 3)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
    
    def forward(self, batch):
        '''
            support feat_type: utt | frm-align | frm-unalign
        '''
        audio_hidden = self.audio_encoder(batch['audios']) # [32, 128]
        text_hidden  = self.text_encoder(batch['texts'])   # [32, 128]
        video_hidden = self.video_encoder(batch['videos']) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]

        features  = fused_feat.squeeze(axis=2) # [32, 128] => 解决batch=1报错的问题
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
