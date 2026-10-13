"""
paper: Memory Fusion Network for Multi-View Sequential Learning
From: https://github.com/pliang279/MFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFN(nn.Module):
    def __init__(self, args):
        super(MFN, self).__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        self.mem_dim = args.mem_dim
        self.hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        # params: intermedia
        total_h_dim =  self.hidden_dim * 3
        attInShape = total_h_dim * args.window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        output_dim = self.hidden_dim // 2

        # each modality has one lstm cell
        self.lstm_l = nn.LSTMCell(text_dim,  self.hidden_dim)
        self.lstm_a = nn.LSTMCell(audio_dim, self.hidden_dim)
        self.lstm_v = nn.LSTMCell(video_dim, self.hidden_dim)

        self.att1_fc1 = nn.Linear(attInShape, self.hidden_dim)
        self.att1_fc2 = nn.Linear(self.hidden_dim, attInShape)
        self.att1_dropout = nn.Dropout(dropout)

        self.att2_fc1 = nn.Linear(attInShape, self.hidden_dim)
        self.att2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.att2_dropout = nn.Dropout(dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma1_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(dropout)

        self.out_fc1 = nn.Linear(final_out, self.hidden_dim)
        self.out_fc2 = nn.Linear(self.hidden_dim, output_dim)
        self.out_dropout = nn.Dropout(dropout)

        # output results
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)
    

    # MFN needs aligned multimodal features
    def forward(self, batch):
        
        '''
        simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        audio_x: tensor of shape (batch, seqlen, audio_in)
        video_x: tensor of shape (batch, seqlen, video_in)
        text_x: tensor of shape  (batch, seqlen, text_in)
        '''
        assert batch['audios'].size()[1] == batch['videos'].size()[1]
        assert batch['audios'].size()[1] == batch['texts'].size()[1]

        text_x  = batch['texts'].permute(1,0,2)  # [seqlen, batch, dim]
        audio_x = batch['audios'].permute(1,0,2) # [seqlen, batch, dim]
        video_x = batch['videos'].permute(1,0,2) # [seqlen, batch, dim]

        # x is t x n x d
        n = text_x.size()[1] # n = batch
        t = text_x.size()[0] # t = seqlen
        self.h_l = torch.zeros(n, self.hidden_dim).cuda()
        self.h_a = torch.zeros(n, self.hidden_dim).cuda()
        self.h_v = torch.zeros(n, self.hidden_dim).cuda()
        self.c_l = torch.zeros(n, self.hidden_dim).cuda()
        self.c_a = torch.zeros(n, self.hidden_dim).cuda()
        self.c_v = torch.zeros(n, self.hidden_dim).cuda()
        self.mem = torch.zeros(n, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t): # lstm 中每个step单独处理

            # prev time step [这里的 c 指的就是 lstm 里面的 cell state]
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v

            # curr time step
            new_h_l, new_c_l = self.lstm_l(text_x[i],  (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))
            
            # concatenate and attention
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs  = torch.cat([new_c_l, new_c_a, new_c_v],  dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention * cStar
            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)

            # update (hidden, cell) in lstm
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v

            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h [就是一个逐步交互的过程]
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        features = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        self.last_hs = last_hs # for outside loading

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
    