"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

import torch
import torch.nn as nn
from torch.autograd import Function

from .modules.encoder import MLPEncoder, LSTMEncoder

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class MISA(nn.Module):
    def __init__(self, args):
        super(MISA, self).__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout           # 0.3
        hidden_dim = args.hidden_dim     # 128
        self.sim_weight = args.sim_weight     # 1.0
        self.diff_weight = args.diff_weight   # 0.3
        self.recon_weight = args.recon_weight # 1.0
        self.grad_clip = args.grad_clip
        
        # params: intermedia
        output_dim = hidden_dim // 2

        # modality-specific encoder
        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # map into a common space
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_t.add_module('project_t_activation', nn.ReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(hidden_dim))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_v.add_module('project_v_activation', nn.ReLU())
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(hidden_dim))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_a.add_module('project_a_activation', nn.ReLU())
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(hidden_dim))

        # private encoders
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        # reconstruct
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

        # fusion + cls
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=hidden_dim*6, out_features=hidden_dim*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=hidden_dim*3, out_features=output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,  nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

        
    def reconstruct(self):
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    ##########################################################
    ## inter loss calculation
    ##########################################################
    def get_recon_loss(self):
        loss =  MSE()(self.utt_t_recon, self.utt_t_orig)
        loss += MSE()(self.utt_v_recon, self.utt_v_orig)
        loss += MSE()(self.utt_a_recon, self.utt_a_orig)
        loss = loss / 3.0
        return loss

    def get_diff_loss(self):
        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        # Between private and shared
        loss =  DiffLoss()(private_t, shared_t)
        loss += DiffLoss()(private_v, shared_v)
        loss += DiffLoss()(private_a, shared_a)

        # Across privates
        loss += DiffLoss()(private_a, private_t)
        loss += DiffLoss()(private_a, private_v)
        loss += DiffLoss()(private_t, private_v)
        return loss

    def get_cmd_loss(self):
        # losses between shared states
        loss =  CMD()(self.utt_shared_t, self.utt_shared_v, 5)
        loss += CMD()(self.utt_shared_t, self.utt_shared_a, 5)
        loss += CMD()(self.utt_shared_a, self.utt_shared_v, 5)
        loss = loss/3.0
        return loss
    
    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            text_feat:  tensor of shape (batch, seqlen2, text_in)
            video_feat: tensor of shape (batch, seqlen3, video_in)
        '''
        utterance_audio = self.audio_encoder(batch['audios']) # [batch, hidden]
        utterance_text  = self.text_encoder(batch['texts'])   # [batch, hidden]
        utterance_video = self.video_encoder(batch['videos']) # [batch, hidden]

        # shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        features = self.fusion(h)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = self.diff_weight  * self.get_diff_loss() + \
                    self.sim_weight   * self.get_cmd_loss()  + \
                    self.recon_weight * self.get_recon_loss()

        return features, emos_out, vals_out, interloss
