import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from utils import *

# class GraphConvolution(Module):
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.sparse.mm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, conv_name='gcn', n_heads=1):
        super(GraphConvolution, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_features, out_features)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_features, out_features // n_heads, heads=n_heads)
        elif self.conv_name == 'sage':
            self.base_conv = SAGEConv(in_features, out_features)
        else:
            print("no predefined conv layer {} !".format(conv_name))
        # nn.init.xavier_uniform_(self.base_conv)

    def forward(self, input_x, edge_index):
        if self.conv_name == 'gcn':
            return self.base_conv(input_x, edge_index)
        elif self.conv_name == 'gat':
            # return self.base_conv(input_x, edge_index)
            ##return_attention_weights=True  If set to True, will additionally return the tuple (edge_index, attention_weights)
            #out, (edge_index, alpha)
            out, b =self.base_conv(input_x, edge_index, return_attention_weights=True) ##返回alpha参数
            _, attention_weights = b
            print(f'    对年月日week的注意力系数：{attention_weights[[-8,-6,-4,-2]]}')
            return out
        elif self.conv_name == 'sage':
        	return self.base_conv(input_x, edge_index)


TIME_NUM_DICT = {
    MOVIE : 94,
    YELP_STAR : 65
}

TIME_DIM_NUM_DICT = {
    MOVIE : 5,
    YELP_STAR : 4
}

'''
@:param: embeddings          env 中 user + item embedding
@:param: layers=1            GNN的层数
@:param: rnn_layer=1         transformer 的 层数
@:param: conv_name='gat'     GNN的类别
@:param: n_heads = 1         注意力头数目
@:param: fix_emb = True     nn.Embedding(uif) 与 self.time_embedding(time 16) 是否固定
'''
class GraphEncoder(Module):
    def __init__(self, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=1, rnn_layer=1, conv_name='gat', n_heads = 1, time_emb_16 = None, data_name = MOVIE, seq_conv_name = 'gcn', seq_conv_layer = 1, time_to_hidden = None):
        super(GraphEncoder, self).__init__()
        self.data_name = data_name
        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)

        self.time_embedding = nn.Embedding(TIME_NUM_DICT[self.data_name], time_emb_16.shape[1])
        if time_emb_16 is not None:
            print(f"pre-trained embeddings-16 with time")
            self.time_embedding = self.time_embedding.from_pretrained(time_emb_16, freeze=fix_emb)

        # 16转64  为学习的参数
        self.time_transfer = Parameter(torch.FloatTensor(time_emb_16.shape[1], emb_size))
        nn.init.xavier_uniform_(self.time_transfer)

        self.time_to_hidden = Parameter(time_to_hidden)     # 16dim时间拼接起来80dim后 转 dim

        self.time80_to_hidden = Parameter(torch.FloatTensor(self.time_to_hidden.shape[0], hidden_size))  # 拼接起来后 80*100
        nn.init.xavier_uniform_(self.time80_to_hidden)

        self.twohidden_to_onehidden = Parameter(torch.FloatTensor(2*hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.twohidden_to_onehidden)

        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.fea_num = len(kg.G['feature'])
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            print(f'conv_name = {conv_name};  n_heads = {n_heads};  layers = {layers}')
            for l in range(layers):
                self.gnns.append(GraphConvolution(indim, outdim, conv_name=conv_name, n_heads=n_heads))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

        ''' 2. seq 的 gcn网络   '''
        self.seq_gnns = nn.ModuleList()
        print(f'seq_gnns conv_name = {seq_conv_name};  seq_conv_layer = {seq_conv_layer}')
        indim, outdim = emb_size, hidden_size
        for l in range(seq_conv_layer):
            self.seq_gnns.append(GraphConvolution(indim, outdim, conv_name=seq_conv_name))
            indim = outdim

        '''trans for seq'''
        self.transformer_for_seq = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400),
            num_layers=rnn_layer)

        self.fc1_for_seq = nn.Linear(hidden_size, hidden_size)

        self.habit_cur_time_score = nn.Linear(2*hidden_size, hidden_size)       # 用来计算long interest 的weight

    # 输入是state   ## 通过env 里面 _get_state 获得的 state
    def forward(self, b_state):
        """
        :param b_state [N]  输入的state是许多dict
        :return: [N x L x d]
        """
        batch_output = []
        batch_output_seq_graph = []     #TODO: 注意 nan了
        batch_long_interest_weight = []
        for s in b_state:   #对每一个状态进行遍历
            #  neighbors = cur_node + user + cand_items + reachable_feature +   self.time_node
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(self.device)    #neighbors：LongTensor   adj: LongTensor
            # uif full_id    times short_id     type都为LongTensor 接下来要过nn.Embedding
            uifs,times = neighbors[:-TIME_DIM_NUM_DICT[self.data_name]].to(self.device),  (neighbors[-TIME_DIM_NUM_DICT[self.data_name]:]-(s['user_length'] + s['item_length'] + s['feature_length'])).to(self.device)  # acc_att, user, cand_items, reachable_feature

            cur_time_emb16 = self.time_embedding(times)  ### 当前时间的time emb16   是给seq graph用的
            cur_time_emb100 = torch.matmul(cur_time_emb16.view(1, -1),self.time80_to_hidden).squeeze()   #给seq graph用的

            uifs_emb = self.embedding(uifs)
            time_emb = torch.matmul(self.time_embedding(times),self.time_transfer)   #time 从16 转为64维

            input_state = torch.cat((uifs_emb, time_emb), dim=0)
            if self.gcn:
                for index, gnn in enumerate(self.gnns):
                    print(f'第{index+1}层GNN')
                    output_state = gnn(input_state, adj)
                    input_state = output_state
                batch_output.append(output_state)
            else:
                output_state = F.relu(self.fc2(input_state))
                batch_output.append(output_state)


            #############################   seq graph 的处理

            seq_neighbors, seq_adj = s['seq_neighbors'].to(self.device), s['seq_edge_index'].to(self.device)
            seq_item_node_full_id_len, seq_fea_node_len = s['seq_item_node_full_id_len'], s['seq_fea_node_len']
            seq_graph_item_att_index = seq_neighbors[:seq_item_node_full_id_len+seq_fea_node_len] #节点提取出来
            seq_graph_uif_emb = self.embedding(seq_graph_item_att_index)            #  item fea 的 emb

            seq_full_index_map_time_tuple = s['seq_full_index_map_time_tuple']
            time_index = seq_neighbors[seq_item_node_full_id_len+seq_fea_node_len:].cpu().tolist()  #time的索引

            # 创建一个空列表，用于存储每次循环得到的 Tensor
            tensor_list = []

            #### 每个索引变成对应的embedding
            for cur_time_index in time_index:
                time_node_tuple = seq_full_index_map_time_tuple[cur_time_index]
                time_node_tuple_long_tensor = torch.tensor(time_node_tuple, dtype=torch.long)
                time_node_tuple_long_tensor = time_node_tuple_long_tensor - (s['user_length'] + s['item_length'] + s['feature_length'])
                time_node_tuple_long_tensor = time_node_tuple_long_tensor.to(self.device)
                time_emb16 = self.time_embedding(time_node_tuple_long_tensor)
                time_emb16_flatten = time_emb16.view(1, -1)
                # 将展平后的 Tensor添加到列表中
                tensor_list.append(time_emb16_flatten)

            if len(tensor_list) == 0:
                print('error!')

            seq_graph_time_node_emb_80dim = torch.cat(tensor_list, dim=0)
            seq_graph_time_node_emb_64dim = torch.matmul(seq_graph_time_node_emb_80dim,self.time_to_hidden)      # seq graph 图的time embedding

            seq_graph_input_state = torch.cat((seq_graph_uif_emb, seq_graph_time_node_emb_64dim), dim=0)

            for index, gnn in enumerate(self.seq_gnns):
                print(f'seq_grah 第{index+1}层GNN')
                output_state = gnn(seq_graph_input_state, seq_adj)
                seq_graph_input_state = output_state
            # batch_output_seq_graph.append(output_state)

            ###################   将时间信息 高斯分布
            #  长期兴趣  时间信息   均值
            long_interset_emb = output_state[:seq_item_node_full_id_len]   # item 序列 emb           长期兴趣emb
            habit_time_emb = output_state[seq_item_node_full_id_len+seq_fea_node_len:]    # 作为习惯的time emb
            # long_interset_emb = torch.mean(long_interset_emb, dim=0)

            # 时间差
            habit_time_emb = torch.mean(habit_time_emb, dim=0)
            score_input = torch.cat([cur_time_emb100,habit_time_emb])
            # cha_time_emb = cur_time_emb100 - habit_time_emb
            long_interset_weight = F.sigmoid(self.habit_cur_time_score(score_input))     # sigmod 会在最开始训练时出现只有0 / 1   会造成梯度消失  --->   sigmod

            # final_long_interset_emb = long_interset_emb * long_interset_weight
            # batch_output_seq_graph.append(final_long_interset_emb)

            batch_long_interest_weight.append(long_interset_weight)
            batch_output_seq_graph.append(long_interset_emb)

        seq_embeddings = []
        for s, o in zip(b_state, batch_output):
            dddd = o[:len(s['cur_node'])+1,:][None,:]
            seq_embeddings.append(dddd)   #acc_fea and user  TransPart

        seq_graph_final_add_embeddings = []
        for s,o in zip(b_state, batch_output_seq_graph):
            seq_graph_final_add_embeddings.append(o[:,:][None,:])

        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
            seq_graph_final_add_embeddings = self.padding_seq(seq_graph_final_add_embeddings)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]
        seq_graph_final_add_embeddings = torch.cat(seq_graph_final_add_embeddings, dim=0)

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1,0,2) #[N*1*D]
        elif self.seq == 'transformer':
            seq_embeddings = torch.mean(self.transformer(seq_embeddings), dim=1, keepdim=True)
        elif self.seq == 'mean':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)

        seq_embeddings = F.relu(self.fc1(seq_embeddings))       # dim = 100

        ## 长期兴趣过trans
        seq_graph_final_add_embeddings = torch.mean(self.transformer_for_seq(seq_graph_final_add_embeddings), dim=1, keepdim=True)   # batch * 1 * dim

        seq_graph_final_add_embeddings = F.relu(self.fc1_for_seq(seq_graph_final_add_embeddings))   # batch * 1 * dim

        seq_graph_final_add_embeddings_weight = torch.stack(batch_long_interest_weight).unsqueeze(1)    #weight
        seq_graph_final_add_embeddings = seq_graph_final_add_embeddings * seq_graph_final_add_embeddings_weight

        long_short_interest = torch.cat([seq_embeddings,seq_graph_final_add_embeddings], dim=-1)
        result_embeddings = torch.matmul(long_short_interest, self.twohidden_to_onehidden)
        # result_embeddings = seq_embeddings + seq_graph_final_add_embeddings
        return result_embeddings
    
    
    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq
