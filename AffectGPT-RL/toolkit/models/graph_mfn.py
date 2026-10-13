"""
paper: Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph
Reference From: https://github.com/pliang279/MFN & https://github.com/A2Zadeh/CMU-MultimodalSDK
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations

class DynamicFusionGraph(nn.Module):
    def __init__(self, pattern_model, in_dimensions, out_dimension, efficacy_model):
        """
        Args:
            pattern_model - nn.Module, a nn.Sequential model which will be used as core of the models inside the DFG
            in_dimensions - List, input dimensions of each modality
            out_dimension - int,  output dimension of the pattern models
            efficacy_model - the core of the efficacy model
        """
        super(DynamicFusionGraph, self).__init__()
        self.num_modalities = len(in_dimensions)
        self.in_dimensions = in_dimensions
        self.out_dimension = out_dimension

        # in this part we sort out number of connections, how they will be connected etc.
        # powerset = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)] for three modal fusion. 
        self.powerset = list(
            chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities + 1)))[1:]
        
        # initializing the models inside the DFG
        self.input_shapes = {tuple([key]): value for key, value in zip(range(self.num_modalities), in_dimensions)}
        
        self.networks = {}
        self.total_input_efficacies = 0 # total_input_efficacies: for alpha list size = [batch_size, total_input_efficacies].
        
        # loop over n-modal node (n >= 2)
        for key in self.powerset[self.num_modalities:]:
            # connections coming from the unimodal components
            unimodal_dims = 0
            for modality in key:
                unimodal_dims += in_dimensions[modality]
            multimodal_dims = ((2 ** len(key) - 2) - len(key)) * out_dimension
            self.total_input_efficacies += 2 ** len(key) - 2
            # for the network that outputs key component, what is the input dimension
            final_dims = unimodal_dims + multimodal_dims
            self.input_shapes[key] = final_dims
            pattern_copy = copy.deepcopy(pattern_model)

            # final_model: transform the input to the node into out_dimension dim.
            final_model = nn.Sequential(
                *[nn.Linear(self.input_shapes[key], list(pattern_copy.children())[0].in_features), pattern_copy]).cuda()
            self.networks[key] = final_model

        # finished construction weights, now onto the t_network which summarizes the graph
        self.total_input_efficacies += 2 ** self.num_modalities - 1

        self.t_in_dimension = unimodal_dims + (2 ** self.num_modalities - (self.num_modalities) - 1) * out_dimension
        pattern_copy = copy.deepcopy(pattern_model)

        # self.t_network: generate top level representation τ 
        self.t_network = nn.Sequential(*[nn.Linear(self.t_in_dimension, list(pattern_copy.children())[0].in_features), pattern_copy]).cuda()

        # self.efficacy_model: generate the alpha list using the singleton vertex input. 
        # (in 3 modal [batch_size, dim_l+dim_v+dim_a] -> [batch_size, 19])
        self.efficacy_model = nn.Sequential(
            *[nn.Linear(sum(in_dimensions), list(efficacy_model.children())[0].in_features), efficacy_model,
              nn.Linear(list(efficacy_model.children())[-1].out_features, self.total_input_efficacies)]).cuda()

    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):

        outputs = {}
        for modality, index in zip(in_modalities, range(len(in_modalities))):
            outputs[tuple([index])] = modality
        efficacies = self.efficacy_model(torch.cat([x for x in in_modalities], dim=1))
        efficacy_index = 0
        for key in self.powerset[self.num_modalities:]:
            small_power_set = list(chain.from_iterable(combinations(key, r) for r in range(len(key) + 1)))[1:-1]
            this_input = torch.cat([outputs[x] * efficacies[:, efficacy_index + y].view(-1, 1) for x, y in
                                    zip(small_power_set, range(len(small_power_set)))], dim=1)
            
            outputs[key] = self.networks[key](this_input)
            efficacy_index += len(small_power_set)

        small_power_set.append(tuple(range(self.num_modalities)))
        t_input = torch.cat([outputs[x] * efficacies[:, efficacy_index + y].view(-1, 1) for x, y in
                                zip(small_power_set, range(len(small_power_set)))], dim=1)
        t_output = self.t_network(t_input)
        return t_output, outputs, efficacies

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)


class Graph_MFN(nn.Module):
    def __init__(self, args):
        super(Graph_MFN, self).__init__()
        
        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        self.mem_dim = args.mem_dim
        self.dropout = args.dropout
        self.hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        # params: intermedia params
        total_h_dim = self.hidden_dim * 3
        gammaInShape = self.hidden_dim + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        output_dim = self.hidden_dim // 2

        self.lstm_l = nn.LSTMCell(text_dim,  self.hidden_dim)
        self.lstm_a = nn.LSTMCell(audio_dim, self.hidden_dim)
        self.lstm_v = nn.LSTMCell(video_dim, self.hidden_dim)

        # Here Changed! Todo : add Arg param singleton_l singleton_a singleton_v
        self.l_transform = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.a_transform = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_transform = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Here Changed! (initialize the DFG part) Todo : add Arg param inner node dimension.
        pattern_model = nn.Sequential(nn.Linear(100, self.hidden_dim)).cuda()
        efficacy_model = nn.Sequential(nn.Linear(100, self.hidden_dim)).cuda() # Note : actually here inner_node_dim can change arbitrarily 
        self.graph_mfn = DynamicFusionGraph(pattern_model, [self.hidden_dim, self.hidden_dim, self.hidden_dim], self.hidden_dim, efficacy_model).cuda()

        # Here Changed! (alter the dim param.)
        self.att2_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim) # Note: might (inner_node_dim = self.mem_dim) is a common choice.
        self.att2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.att2_dropout = nn.Dropout(self.dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma1_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(self.dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(self.dropout)

        self.out_fc1 = nn.Linear(final_out, self.hidden_dim)
        self.out_fc2 = nn.Linear(self.hidden_dim, output_dim)
        self.out_dropout = nn.Dropout(self.dropout)

        # output results
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)
    
    # 整体结构和MFN基本一样，但是把MFN中的attention换成了graph-based fusion
    def forward(self, batch):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)
            video_x: tensor of shape (batch_size, sequence_len, video_in)
            text_x:  tensor of shape (batch_size, sequence_len, text_in)
        '''
        assert batch['audios'].size()[1] == batch['videos'].size()[1]
        assert batch['audios'].size()[1] == batch['texts'].size()[1]

        text_x  = batch['texts'].permute(1,0,2)
        audio_x = batch['audios'].permute(1,0,2)
        video_x = batch['videos'].permute(1,0,2)

        # x is t x n x d
        n = text_x.size()[1]
        t = text_x.size()[0]
        self.h_l = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.h_a = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.h_v = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.c_l = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.c_a = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.c_v = torch.zeros(n, self.hidden_dim).to(text_x.device)
        self.mem = torch.zeros(n, self.mem_dim).to(text_x.device)
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_mems = []
        all_efficacies = []

        for i in range(t):
            # prev time step (Here Changed !)
            prev_h_l = self.h_l
            prev_h_a = self.h_a
            prev_h_v = self.h_v

            # curr time step
            new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))
            
            # concatenate (Here Changed!)
            l_input = torch.cat([prev_h_l, new_h_l], dim=1)
            l_singleton_input = F.relu(self.l_transform(l_input))
            a_input = torch.cat([prev_h_a, new_h_a], dim=1)
            a_singleton_input = F.relu(self.a_transform(a_input))
            v_input = torch.cat([prev_h_v, new_h_v], dim=1)
            v_singleton_input = F.relu(self.v_transform(v_input))

            # Note: we might want to record the efficacies for some reasons.
            attended, _, efficacies = self.graph_mfn([l_singleton_input, a_singleton_input, v_singleton_input])
            all_efficacies.append(efficacies.cpu().detach().squeeze().numpy())

            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1 * self.mem + gamma2 * cHat
            all_mems.append(self.mem)

            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs  = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        features = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
       
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss