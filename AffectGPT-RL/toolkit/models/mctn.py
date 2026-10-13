import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.encoder import LSTMEncoder

class Encoder(nn.Module): # Encoder
    def __init__(self, input_dim, hidden_dim, dropout, depth, bidirectional=True, lengths=None):
        super().__init__()
        
        if depth == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        self.hidden_dim = hidden_dim
        self.bidirectional=bidirectional
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=depth, dropout=rnn_dropout, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias = False)

    def forward(self, x, lengths): 
        '''
        x : (batch_size, sequence_len, in_size)
        '''
        enc_output, enc_state = self.rnn(x)
        if self.bidirectional:
            h = self.dropout(torch.add(enc_output[:,:,:self.hidden_dim],enc_output[:,:,self.hidden_dim:]))
        else:
            h = self.dropout(enc_state[0].squeeze())
        join = h
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hidden_dim]
        s = torch.tanh(self.fc(torch.add(enc_state[0][-1],enc_state[0][-2]))) ####
        
        return join, s


class Attention(nn.Module): # Attention layer of decoder
    def __init__(self, hidden_dim, lengths=None):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias = False)
        
    def forward(self, s, join):
        src_len = join.shape[0]
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hidden_dim]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        join = join.transpose(0, 1)
        # energy = [batch_size, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((s, join), dim = 2)))
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module): # Seq2Seq with attention
    def __init__(self, encoder, decoder, device, lengths=None):
        super().__init__()
        self.lengths=lengths
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5): 
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)   
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src, self.lengths)   
        dec_input = trg[0,:]
        for t in range(1, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)
            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = trg[t,:]
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return enc_output, outputs


class Decoder(nn.Module): # Decoder
    def __init__(self, output_dim, hidden_dim, dropout, depth, attention, bidirectional, lengths=None):
        super().__init__()

        if depth == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.attention = attention
        self.hidden_dim=hidden_dim
        self.rnn = nn.LSTM(output_dim+hidden_dim, hidden_dim, num_layers=depth, dropout=rnn_dropout, bidirectional = self.bidirectional)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, s, join):
        dec_input = dec_input.unsqueeze(1).transpose(0, 1) 
        a = self.attention(s, join).unsqueeze(1)
        # join
        join = join.transpose(0, 1)
        c = torch.bmm(a, join).transpose(0, 1)
        rnn_input = torch.cat((dec_input, c), dim = 2)
        dec_output, dec_state = self.rnn(rnn_input) 

        if self.bidirectional:
            dec_output = torch.add(dec_output[:,:,:self.hidden_dim],dec_output[:,:,self.hidden_dim:])
            h = torch.add(dec_state[0][-1],dec_state[0][-2])

        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        pred = self.fc_out(torch.cat((dec_output, c), dim = 1))
        
        return pred, h.squeeze(0)


class MCTN(nn.Module):
    def __init__(self, args):
        super().__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout # 0
        hidden_dim = args.hidden_dim # 32
        self.loss_weight = args.loss_weight
        self.teacher_forcing_ratio = args.teacher_forcing_ratio # 0.5
        self.grad_clip = args.grad_clip

        # params: intermedia
        depth = [1, 1]
        self.output_dim = max(audio_dim, text_dim, video_dim)
        
        self.attn1=Attention(hidden_dim)
        self.encoder1=Encoder(self.output_dim, hidden_dim, dropout, depth[0])
        self.decoder1=Decoder(self.output_dim, hidden_dim, dropout, depth[1], self.attn1, bidirectional=True)
        self.seq2seq1=Seq2Seq(self.encoder1, self.decoder1, 'cuda', lengths=None)

        self.attn2=Attention(hidden_dim)
        self.encoder2=Encoder(hidden_dim, hidden_dim, dropout, depth[0])
        self.decoder2=Decoder(self.output_dim, hidden_dim, dropout, depth[1], self.attn2, bidirectional=True)
        self.seq2seq2=Seq2Seq(self.encoder2, self.decoder2, 'cuda', lengths=None)
        
        # output results
        self.fc_out_0 = LSTMEncoder(hidden_dim, hidden_dim, dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        '''
        # 感觉用aligned feature能够方便计算重建loss吧
        text_feat:  [batch, seqlen, feat1]
        audio_feat: [batch, seqlen, feat2]
        video_feat: [batch, seqlen, feat3]
        '''
        assert batch['audios'].size()[1] == batch['videos'].size()[1]
        assert batch['audios'].size()[1] == batch['texts'].size()[1]

        # pad into fixed length
        maxn = self.output_dim
        text   = F.pad(batch['texts'],  (0, maxn - len(batch['texts'][0][0])))
        audio  = F.pad(batch['audios'], (0, maxn - len(batch['audios'][0][0])))
        vision = F.pad(batch['videos'], (0, maxn - len(batch['videos'][0][0])))  
        
        join, video_1 = self.seq2seq1(text,    vision, self.teacher_forcing_ratio)
        _,     text_1 = self.seq2seq1(video_1, text,   self.teacher_forcing_ratio)
        join, audio_1 = self.seq2seq2(join,    audio,  self.teacher_forcing_ratio)
        
        features  = self.fc_out_0(join)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)

        # cycle-reconstruction loss
        loss_v = nn.MSELoss()(video_1,vision)
        loss_t = nn.MSELoss()(text_1, text)
        loss_a = nn.MSELoss()(audio_1,audio)
        interloss = self.loss_weight * (loss_v + loss_t + loss_a)
        
        return features, emos_out, vals_out, interloss

  




