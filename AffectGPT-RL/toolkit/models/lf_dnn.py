"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
Description: unimodal encoder + concat fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.encoder import MLPEncoder, LSTMEncoder

class LF_DNN(nn.Module):
    def __init__(self, args):
        super(LF_DNN, self).__init__()
        
        text_dim    = args.text_dim
        audio_dim   = args.audio_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        
        # define the pre-fusion subnetworks
        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim)
        self.post_fusion_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)


    # modality-specific mlp + concat + mlp + cls
    def forward(self, batch):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x:  tensor of shape (batch_size, text_in)
        '''
        audio_h = self.audio_encoder(batch['audios'])
        video_h = self.video_encoder(batch['videos'])
        text_h  = self.text_encoder(batch['texts'])

        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        features = F.relu(self.post_fusion_layer_2(x), inplace=True)
        
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss

       
        