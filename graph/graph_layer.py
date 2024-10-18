""""
Define a generic GRM layer model
"""
from pickletools import decimalnl_short
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph_util import *
#from graph.coco_data import *
from omegaconf import OmegaConf
#from .global_settings import GPU_ID
from torch.autograd import Variable
from torch.nn import Parameter
import math
#from .init_weights import init_weights

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

cuda_suffix = 'cuda:' + str(GPU_ID) if len(str(GPU_ID)) == 1 else "cuda"
device = torch.device(cuda_suffix if torch.cuda.is_available() else "cpu")





#Graph Reasoning Module
#Graph convolution
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight.cuda())
        #support = torch.matmul(input, self.weight)
        adj=adj.to(input.cuda())
        #adj=adj.to(input.cpu())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



#overall model layer
class GraphLearner(nn.Module):

    def __init__(self, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(GraphLearner, self).__init__()


        self.graph_reasoning1 = GraphConvolution(512,128)
        self.graph_reasoning2 = GraphConvolution(128,256)
        self.graph_adj_mat = torch.FloatTensor(graph_adj_mat).cuda()
        self.fc_in = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 512)
        
    def forward(self, fg_text_features, adj):
        #[？，M, H*W]
        #x = self.conv1(x)
        graph_norm_adj = adj

        #print("#############fg_text_features2",fg_text_features.shape)
        node = fg_text_features
        
        fg_text_features = fg_text_features.type(graph_norm_adj.dtype)#.detach()
        node_out = self.fc_in(fg_text_features)
      
        graph_norm_adj = normalize_adjacency(graph_norm_adj)
        
        #fg_text_features= F.dropout(fg_text_features, 0.3)
        text_node1 = self.graph_reasoning1(fg_text_features, graph_norm_adj)#.detach()
        text_node1 = F.relu(text_node1)
        
        text_node = F.dropout(text_node1, 0.3)
        text_node = self.graph_reasoning2(text_node, graph_norm_adj)#.detach()
        text_node = F.relu(text_node)

        text_node = text_node.type(node.dtype).detach()
        text_node = text_node + node_out
           

        return text_node


class GraphFusion(nn.Module):
    def __init__(self, num_state, num_node):
    #def __init__(self, inp, vis_node):
        super(GraphFusion, self).__init__()
        #num_state=256
        
        self.vis_gcn = GCN(num_state, num_node)
        self.word_gcn = GCN(num_state, num_node)
        self.transfer = GraphTransfer(num_state)
        self.gamma_vis = nn.Parameter(torch.zeros(num_node))
        self.gamma_word = nn.Parameter(torch.zeros(num_node))

    def forward(self, inp, vis_node):
        #inp = self.inp
        #print("##########inp:", inp.shape)
        inp = inp.transpose(1,0).unsqueeze(0).float()
        #print("##########inp2:", inp.shape)
        vis_node = vis_node.transpose(1,0).unsqueeze(0).float()
        inp = self.word_gcn(inp)
        new_V = self.vis_gcn(vis_node)
        #new_V = vis_node
        class_node, vis_out = self.transfer(inp, new_V)

        class_node =  inp + class_node
        new_V = vis_out + new_V
  
        return class_node.squeeze(0).transpose(1,0).detach(), new_V.squeeze(0).transpose(1,0).detach()

class GCN(nn.Module):
    def __init__(self, num_state=256, num_node=20, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(
            num_node,
            num_node,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            num_state,
            num_state,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class GraphTransfer(nn.Module):
    def __init__(self, in_dim):
        super(GraphTransfer, self).__init__()
        self.channle_in = in_dim
        self.query_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        
        self.value_conv_vis = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_word = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax_vis = nn.Softmax(dim=-1)
        self.softmax_word = nn.Softmax(dim=-2)

    def forward(self, word, vis_node):
        m_batchsize, C, Nc = word.size()
        m_batchsize, C, Nn = vis_node.size()

        proj_query = self.query_conv(word).view(m_batchsize, -1, Nc).permute(0, 2, 1)
        proj_key = self.key_conv(vis_node).view(m_batchsize, -1, Nn)

        energy = torch.bmm(proj_query, proj_key)
        attention_vis = self.softmax_vis(energy).permute(0, 2, 1)
        attention_word = self.softmax_word(energy)

        proj_value_vis = self.value_conv_vis(vis_node).view(m_batchsize, -1, Nn)
        proj_value_word = self.value_conv_word(word).view(m_batchsize, -1, Nc)

        class_out = torch.bmm(proj_value_vis, attention_vis)
        node_out = torch.bmm(proj_value_word, attention_word)
        return class_out, node_out


def cosine_similarity(x):
    x_norm = F.normalize(x, p=2, dim=-1)
    return torch.matmul(x_norm, x_norm.transpose(0, 1))


def generate_clip_input(img_224, cam_224, label, clip_model, clip_input_size, device):
    batch_size, num_classes = label.size()
    
    # Flatten cam_224 to [batch_size * num_classes, 1, 224, 224]
    cam_224 = cam_224.reshape(batch_size * num_classes, 1, clip_input_size, clip_input_size)
    img_224 = img_224.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).reshape(batch_size * num_classes, 3, clip_input_size, clip_input_size)
  
    fg_224_eval = torch.zeros(batch_size, num_classes, 3, clip_input_size, clip_input_size).cuda()
    bg_224_eval = torch.zeros(batch_size, num_classes, 3, clip_input_size, clip_input_size).cuda()


    for i in range(batch_size):
        for j in range(num_classes):
            if label[i, j] == 1:
                fg_224_eval[i, j] = cam_224[i * num_classes + j] * img_224[i]
                bg_224_eval[i, j] = (1 - cam_224[i * num_classes + j]) * img_224[i]

    fg_224_eval_flat = fg_224_eval.view(batch_size * num_classes, 3, clip_input_size, clip_input_size)
    bg_224_eval_flat = bg_224_eval.view(batch_size * num_classes, 3, clip_input_size, clip_input_size)     
    
    fg_features = clip_model.encode_image(fg_224_eval_flat).view(batch_size, num_classes, -1).detach()
    bg_features = clip_model.encode_image(bg_224_eval_flat).view(batch_size, num_classes, -1).detach()
  
    return fg_features, bg_features


if __name__ == "__main__":
   pass