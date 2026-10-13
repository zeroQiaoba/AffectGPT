"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
Description: concat + lstm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EF_LSTM(nn.Module):
    def __init__(self, args):
        super(EF_LSTM, self).__init__()

        # params: args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        num_layers = args.num_layers
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        # params: combination
        in_size = text_dim + audio_dim + video_dim
  
        self.lstm = nn.LSTM(in_size, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # output results
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
        

    def forward(self, batch):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_dim)
            video_x: tensor of shape (batch_size, sequence_len, video_dim)
            text_x:  tensor of shape (batch_size, sequence_len, text_dim )
        '''
        assert batch['audios'].size()[1] == batch['videos'].size()[1]
        assert batch['audios'].size()[1] == batch['texts'].size()[1]

        x = torch.cat([batch['texts'], batch['audios'], batch['videos']], dim=-1)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze())
        x = F.relu(self.linear(x), inplace=True)
        features = self.dropout(x)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss

        