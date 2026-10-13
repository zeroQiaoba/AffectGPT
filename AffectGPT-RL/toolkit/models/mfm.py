'''
title: LEARNING FACTORIZED MULTIMODAL REPRESENTATIONS (ICLR19)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .mfn import MFN

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def loss_MMD(zy):
    zy_real_gauss = Variable(torch.randn(zy.size())) # no need to be the same size

    #if args.cuda:
    zy_real_gauss = zy_real_gauss.cuda()
    zy_real_kernel = compute_kernel(zy_real_gauss, zy_real_gauss)
    zy_fake_kernel = compute_kernel(zy, zy)
    zy_kernel = compute_kernel(zy_real_gauss, zy)
    zy_mmd = zy_real_kernel.mean() + zy_fake_kernel.mean() - 2.0*zy_kernel.mean()
    return zy_mmd

class encoderLSTM(nn.Module):
    def __init__(self, d, h): #, n_layers, bidirectional, dropout):
        super(encoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(d, h)
        self.fc1 = nn.Linear(h, h)
        self.h = h

    def forward(self, x):
        # x is t x n x h
        t = x.shape[0]
        n = x.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        all_hs = []
        all_cs = []
        for i in range(t):
            self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
        # last hidden layer last_hs is n x h
        last_hs = all_hs[-1]
        last_hs = self.fc1(last_hs)
        return last_hs

class decoderLSTM(nn.Module):
    def __init__(self, h, d):
        super(decoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(h, h)
        self.fc1 = nn.Linear(h, d)
        self.d = d
        self.h = h
        
    def forward(self, hT, t):
        # x is n x d
        n = hT.shape[0]
        h = hT.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        final_hs = []
        all_hs = []
        all_cs = []
        for i in range(t):
            if i == 0:
                self.hx, self.cx = self.lstm(hT, (self.hx, self.cx))
            else:
                self.hx, self.cx = self.lstm(all_hs[-1], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
            final_hs.append(self.hx.view(1,n,h))
        final_hs = torch.cat(final_hs, dim=0)
        all_recons = self.fc1(final_hs)
        return all_recons


class MFM(nn.Module):
    def __init__(self, args):
        super(MFM, self).__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        self.mem_dim = args.mem_dim
        self.dropout = args.dropout
        self.window_dim = args.window_dim
        self.hidden_dim = args.hidden_dim
        self.lda_xl = args.lda_xl
        self.lda_xa = args.lda_xa
        self.lda_xv = args.lda_xv
        self.lda_mmd = args.lda_mmd
        self.grad_clip = args.grad_clip

        # params: generate
        total_h_dim = self.hidden_dim * 3
        self.z_dim  = self.hidden_dim
        self.f_dim  = self.hidden_dim // 2
        last_mfn_size = total_h_dim + self.mem_dim
        self.output_dim = self.hidden_dim // 2

        self.encoder_l = encoderLSTM(text_dim,  self.z_dim)
        self.encoder_a = encoderLSTM(audio_dim, self.z_dim)
        self.encoder_v = encoderLSTM(video_dim, self.z_dim)

        self.decoder_l = decoderLSTM(self.f_dim*2, text_dim)
        self.decoder_a = decoderLSTM(self.f_dim*2, audio_dim)
        self.decoder_v = decoderLSTM(self.f_dim*2, video_dim)

        self.mfn_encoder = MFN(args)

        self.last_to_zy_fc1 = nn.Linear(last_mfn_size, self.z_dim)

        self.zy_to_fy_fc1 = nn.Linear(self.z_dim, self.f_dim)
        self.zy_to_fy_fc2 = nn.Linear(self.f_dim, self.f_dim)
        self.zy_to_fy_dropout = nn.Dropout(self.dropout)

        self.zl_to_fl_fc1 = nn.Linear(self.z_dim,self.f_dim)
        self.zl_to_fl_fc2 = nn.Linear(self.f_dim,self.f_dim)
        self.zl_to_fl_dropout = nn.Dropout(self.dropout)

        self.za_to_fa_fc1 = nn.Linear(self.z_dim,self.f_dim)
        self.za_to_fa_fc2 = nn.Linear(self.f_dim,self.f_dim)
        self.za_to_fa_dropout = nn.Dropout(self.dropout)

        self.zv_to_fv_fc1 = nn.Linear(self.z_dim,self.f_dim)
        self.zv_to_fv_fc2 = nn.Linear(self.f_dim,self.f_dim)
        self.zv_to_fv_dropout = nn.Dropout(self.dropout)

        self.fy_to_y_fc1 = nn.Linear(self.f_dim,self.f_dim)
        self.fy_to_y_fc2 = nn.Linear(self.f_dim,self.output_dim)
        self.fy_to_y_dropout = nn.Dropout(self.dropout)

        # output results
        self.fc_out_1 = nn.Linear(self.output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(self.output_dim, output_dim2)
        

    def forward(self, batch):
        '''
        Args:
            audio: tensor of shape (batch_size, seqlen, audio_in)
            video: tensor of shape (batch_size, seqlen, video_in)
            text:  tensor of shape (batch_size, seqlen, text_in)
        '''
        audio_feat = batch['audios']
        text_feat  = batch['texts']
        video_feat = batch['videos']

        assert audio_feat.size()[1] == video_feat.size()[1]
        assert audio_feat.size()[1] == text_feat.size()[1]

        x_l = text_feat.permute(1,0,2)
        x_a = audio_feat.permute(1,0,2)
        x_v = video_feat.permute(1,0,2)

        # x is t x n x d
        t = x_l.size()[0]
        
        # inputdim -> hiddendim
        zl = self.encoder_l.forward(x_l)
        za = self.encoder_a.forward(x_a)
        zv = self.encoder_v.forward(x_v)
        self.mfn_encoder(batch)
        mfn_last = self.mfn_encoder.last_hs
        zy = self.last_to_zy_fc1(mfn_last)

        # VAE hidden loss
        mmd_loss = loss_MMD(zl) + loss_MMD(za) + loss_MMD(zv) + loss_MMD(zy)
        
        # hiddendim -> hiddendim // 2
        fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
        fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
        fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
        fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))

        # hiddendim // 2 -> hiddendim
        fyfl = torch.cat([fy, fl], dim=1)
        fyfa = torch.cat([fy, fa], dim=1)
        fyfv = torch.cat([fy, fv], dim=1)
        
        # hiddendim -> [seqlen, hiddendim]
        dec_len = t
        x_l_hat = self.decoder_l.forward(fyfl, dec_len)
        x_a_hat = self.decoder_a.forward(fyfa, dec_len)
        x_v_hat = self.decoder_v.forward(fyfv, dec_len)

        # hiddendim -> outputdim
        features = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        
        # 感觉就和VAE很像，计算 hidden features and gaussian loss，然后计算重建 loss
        gen_loss = self.lda_xl * F.mse_loss(x_l_hat, x_l) + self.lda_xa * F.mse_loss(x_a_hat, x_a) + self.lda_xv * F.mse_loss(x_v_hat, x_v)
        interloss = self.lda_mmd * mmd_loss + gen_loss

        return features, emos_out, vals_out, interloss
