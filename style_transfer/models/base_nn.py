import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from style_transfer.config import Config
from pytorch3d.structures.utils import packed_to_list, list_to_padded
import pdb 

class GraphConvClf(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        
        # Graph Convolution Network
        self.gconvs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.fc1 = nn.Linear(dims[-1], 1024)
        self.fc2 = nn.Linear(1024, classes)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        for gconv in self.gconvs:
            verts = F.relu(gconv(verts, edges))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        verts_padded = list_to_padded(verts_packed)
        
        out  = torch.mean(verts_padded, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)        
        return out

    #def get_forward_fcfeats(self, mesh, layername):
    #    #graph_features = {}
    #    verts = mesh.verts_packed()
    #    edges = mesh.edges_packed()
    #    
    #    for gconv in self.gconvs:
    #        verts = F.relu(gconv(verts, edges))
    #        #verts = gconv(verts, edges)
    #        #graph_features[] = verts
    #    
    #    ### VERTS ###
    #    verts_idx = mesh.verts_packed_to_mesh_idx()
    #    verts_size = tuple(verts_idx.unique(return_counts=True)[1])
    #    verts_packed = packed_to_list(verts, verts_size)
    #    verts_padded = list_to_padded(verts_packed)
    #    
    #    out  = torch.mean(verts_padded, 1)
    #    out_fc1 = self.fc1(out)
    #    out_fc2 = self.fc2(F.relu(out_fc1)) 
    #    if layername=='fc1':       
    #        return out_fc1
    #    elif layername=='fc2':
    #        return out_fc2

    def get_forward_feats(self, mesh, layername):
        #graph_features = {}
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        for ii, gconv in enumerate(self.gconvs):
            #verts = F.relu(gconv(verts, edges))
            pre_verts = gconv(verts, edges)
            if layername=='gconv'+str(ii):
                return pre_verts
            verts = F.relu(pre_verts)
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        verts_padded = list_to_padded(verts_packed)
        
        out  = torch.mean(verts_padded, 1)
        out_fc1 = self.fc1(out)
        out_fc2 = self.fc2(F.relu(out_fc1)) 
        if layername=='fc1':       
            return out_fc1
        elif layername=='fc2':
            return out_fc2