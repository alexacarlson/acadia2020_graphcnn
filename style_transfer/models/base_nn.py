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
        #self.gconvs = nn.ModuleList()
        #dims = [input_dim] + hidden_dims
        #for i in range(len(dims)-1):
        #    self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.dims = [input_dim] + hidden_dims
        for i in range(len(self.dims)-1):
            setattr(self, 'gconv{}'.format(i), GraphConv(self.dims[i], self.dims[i+1], init=gconv_init, directed=False))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.dims[i+1]))

        nn1 = 128 #1024
        self.fc1 = nn.Linear(self.dims[-1], nn1)

        #self.fc1 = nn.Linear(dims[-1], 1024)
        #self.fc2 = nn.Linear(1024, classes)
        self.fc_style = nn.Linear(nn1, 3)
        self.fc_semantics = nn.Linear(nn1, 2)
        self.fc_functionality = nn.Linear(nn1, 4)
        self.fc_aesthetics = nn.Linear(nn1, 5)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc_style.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style.bias, 0)
        nn.init.normal_(self.fc_semantics.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_semantics.bias, 0)
        nn.init.normal_(self.fc_functionality.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality.bias, 0)
        nn.init.normal_(self.fc_aesthetics.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        #print('verts shape before gconv: '+str(verts.shape))

        #for gconv in self.gconvs:
        #    verts = F.relu(gconv(verts, edges))
        for i in range(len(self.dims)-1):
            verts = getattr(self, 'gconv{}'.format(i))(verts, edges)
            verts = F.relu(getattr(self, 'norm{}'.format(i))(verts))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        #verts_padded = list_to_padded(verts_packed)
        #out  = torch.mean(verts_padded, 1)
        verts_meaned = [torch.mean(xmesh.unsqueeze(0), dim=1) for xmesh in verts_packed]
        out = torch.cat(verts_meaned, dim=0)
        
        #print('out shape: '+str(out.shape))
        out = F.relu(self.fc1(out))
        out_style = self.fc_style(out)
        out_semantics = self.fc_semantics(out)
        out_functionality = self.fc_functionality(out)
        out_aesthetics = self.fc_aesthetics(out)
        #out = self.fc2(out)   
        # return out
        #print(out_style.shape, out_semantics.shape, out_functionality.shape, out_aesthetics.shape)
        #print("-------------------------------")
        return [out_style, out_semantics, out_functionality, out_aesthetics]

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

    #def get_forward_feats(self, mesh, layername):
    #    #graph_features = {}
    #    verts = mesh.verts_packed()
    #    edges = mesh.edges_packed()
    #    
    #    for ii, gconv in enumerate(self.gconvs):
    #        #verts = F.relu(gconv(verts, edges))
    #        pre_verts = gconv(verts, edges)
    #        if layername=='gconv'+str(ii):
    #            return pre_verts
    #        verts = F.relu(pre_verts)
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

class GraphConvClf_singletask(nn.Module):
    def __init__(self, cfg, numclasses):
        super(GraphConvClf_singletask, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = numclasses #cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        
        # Graph Convolution Network
        #self.gconvs = nn.ModuleList()
        #dims = [input_dim] + hidden_dims
        #for i in range(len(dims)-1):
        #    self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.dims = [input_dim] + hidden_dims
        for i in range(len(self.dims)-1):
            setattr(self, 'gconv{}'.format(i), GraphConv(self.dims[i], self.dims[i+1], init=gconv_init, directed=False))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.dims[i+1]))

        neurdims=256 #512
        self.fc1 = nn.Linear(self.dims[-1], neurdims)
        self.fc2 = nn.Linear(neurdims, classes)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)        
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        #print('verts shape before gconv: '+str(verts.shape))

        #for gconv in self.gconvs:
        #    verts = F.relu(gconv(verts, edges))
        for i in range(len(self.dims)-1):
            verts = getattr(self, 'gconv{}'.format(i))(verts, edges)
            verts = F.relu(getattr(self, 'norm{}'.format(i))(verts))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        #verts_padded = list_to_padded(verts_packed)
        #out  = torch.mean(verts_padded, 1)
        verts_meaned = [torch.mean(xmesh.unsqueeze(0), dim=1) for xmesh in verts_packed]
        out = torch.cat(verts_meaned, dim=0)
        
        #print('out shape: '+str(out.shape))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)   
        return [out]

    
#class GraphConvClf3(nn.Module):    
class GraphConvClf_singlesemclass(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf_singlesemclass, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        
        ## Graph Convolution Network
        #self.gconvs = nn.ModuleList()
        self.dims = [input_dim] + hidden_dims
        #for i in range(len(dims)-1):
        #    self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        for i in range(len(self.dims)-1):
            setattr(self, 'gconv{}'.format(i), GraphConv(self.dims[i], self.dims[i+1], init=gconv_init, directed=False))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.dims[i+1]))

        neurdims = self.dims[-1] #256 #1024
        neurdims2 = 256
        #self.fc1 = nn.Linear(self.dims[-1], neurdims)

        self.fc_style1 = nn.Linear(neurdims, neurdims2)
        self.fc_functionality1 = nn.Linear(neurdims, neurdims2)
        self.fc_aesthetics1 = nn.Linear(neurdims, neurdims2)
        
        self.fc_style2 = nn.Linear(neurdims2, 3)
        self.fc_functionality2 = nn.Linear(neurdims2, 4)
        self.fc_aesthetics2 = nn.Linear(neurdims2, 5)
        
        #nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        #nn.init.constant_(self.fc1.bias, 0)
                
        nn.init.normal_(self.fc_style1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style1.bias, 0)
        nn.init.normal_(self.fc_functionality1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality1.bias, 0)
        nn.init.normal_(self.fc_aesthetics1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics1.bias, 0)
        
        nn.init.normal_(self.fc_style2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style2.bias, 0)
        nn.init.normal_(self.fc_functionality2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality2.bias, 0)
        nn.init.normal_(self.fc_aesthetics2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics2.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        #for gconv in self.gconvs:
        #    verts = F.relu(gconv(verts, edges))
        for i in range(len(self.dims)-1):
            verts = getattr(self, 'gconv{}'.format(i))(verts, edges)
            verts = F.relu(getattr(self, 'norm{}'.format(i))(verts))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        #verts_padded = list_to_padded(verts_packed)
        #out  = torch.mean(verts_padded, 1)
        verts_meaned = [torch.mean(xmesh.unsqueeze(0), dim=1) for xmesh in verts_packed]
        out = torch.cat(verts_meaned, dim=0)
        
        #out = F.relu(self.fc1(out))
        out_style = self.fc_style2(F.relu(self.fc_style1(out)))
        #out_semantics = self.fc_semantics2(F.relu(self.fc_semantics1(out)))
        out_functionality = self.fc_functionality2(F.relu(self.fc_functionality1(out)))
        out_aesthetics = self.fc_aesthetics2(F.relu(self.fc_aesthetics1(out)))
        #out = self.fc2(out)   
        # return out
        return [out_style, out_functionality, out_aesthetics]
    
class GraphConvClf2(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf2, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        
        # Graph Convolution Network
        #self.gconvs = nn.ModuleList()
        #dims = [input_dim] + hidden_dims
        #for i in range(len(dims)-1):
        #    self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.dims = [input_dim] + hidden_dims
        for i in range(len(self.dims)-1):
            setattr(self, 'gconv{}'.format(i), GraphConv(self.dims[i], self.dims[i+1], init=gconv_init, directed=False))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.dims[i+1]))
            #setattr(self, 'norm{}'.format(i), nn.InstanceNorm1d(self.dims[i+1]))

        #self.fc1 = nn.Linear(self.dims[-1], 1024)

        #self.fc1 = nn.Linear(dims[-1], 1024)
        ##self.fc2 = nn.Linear(1024, classes)
        
        nn1 = 256 #512
        self.fc_style1 = nn.Linear(self.dims[-1], nn1)
        self.fc_semantics1 = nn.Linear(self.dims[-1], nn1)
        self.fc_functionality1 = nn.Linear(self.dims[-1], nn1)
        self.fc_aesthetics1 = nn.Linear(self.dims[-1], nn1)
        
        self.fc_style2 = nn.Linear(nn1, 3)
        self.fc_semantics2 = nn.Linear(nn1, 2)
        self.fc_functionality2 = nn.Linear(nn1, 4)
        self.fc_aesthetics2 = nn.Linear(nn1, 5)
        
        #nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        #nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc_style1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style1.bias, 0)
        nn.init.normal_(self.fc_semantics1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_semantics1.bias, 0)
        nn.init.normal_(self.fc_functionality1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality1.bias, 0)
        nn.init.normal_(self.fc_aesthetics1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics1.bias, 0)
        
        nn.init.normal_(self.fc_style2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style2.bias, 0)
        nn.init.normal_(self.fc_semantics2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_semantics2.bias, 0)
        nn.init.normal_(self.fc_functionality2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality2.bias, 0)
        nn.init.normal_(self.fc_aesthetics2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics2.bias, 0)
        
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        #for gconv in self.gconvs:
        #    verts = F.relu(gconv(verts, edges))
        for i in range(len(self.dims)-1):
            verts = getattr(self, 'gconv{}'.format(i))(verts, edges)
            verts = F.relu(getattr(self, 'norm{}'.format(i))(verts))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        #verts_padded = list_to_padded(verts_packed)
        #out  = torch.mean(verts_padded, 1)
        verts_meaned = [torch.mean(xmesh.unsqueeze(0), dim=1) for xmesh in verts_packed]
        out = torch.cat(verts_meaned, dim=0)
        
        #out = F.relu(self.fc1(out))
        out_style = self.fc_style2(F.relu(self.fc_style1(out)))
        out_semantics = self.fc_semantics2(F.relu(self.fc_semantics1(out)))
        out_functionality = self.fc_functionality2(F.relu(self.fc_functionality1(out)))
        out_aesthetics = self.fc_aesthetics2(F.relu(self.fc_aesthetics1(out)))
        #out = self.fc2(out)   
        # return out
        return [out_style, out_semantics, out_functionality, out_aesthetics]
    
class GraphConvClf3(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf3, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        
        # Graph Convolution Network
        #self.gconvs = nn.ModuleList()
        #dims = [input_dim] + hidden_dims
        #for i in range(len(dims)-1):
        #    self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.dims = [input_dim] + hidden_dims
        for i in range(len(self.dims)-1):
            setattr(self, 'gconv{}'.format(i), GraphConv(self.dims[i], self.dims[i+1], init=gconv_init, directed=False))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm1d(self.dims[i+1]))
            #setattr(self, 'norm{}'.format(i), nn.InstanceNorm1d(self.dims[i+1]))
            
        #self.fc1 = nn.Linear(dims[-1], 1024)
        ##self.fc2 = nn.Linear(1024, classes)

        numn = 512 #256 #1024
        numn2 = 256 #128 #256

        self.fc1 = nn.Linear(self.dims[-1], numn)

        self.fc_style1 = nn.Linear(numn, numn2)
        self.fc_semantics1 = nn.Linear(numn, numn2)
        self.fc_functionality1 = nn.Linear(numn, numn2)
        self.fc_aesthetics1 = nn.Linear(numn, numn2)
        
        self.fc_style2 = nn.Linear(numn2, 3)
        self.fc_semantics2 = nn.Linear(numn2, 2)
        self.fc_functionality2 = nn.Linear(numn2, 4)
        self.fc_aesthetics2 = nn.Linear(numn2, 5)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
                
        nn.init.normal_(self.fc_style1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style1.bias, 0)
        nn.init.normal_(self.fc_semantics1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_semantics1.bias, 0)
        nn.init.normal_(self.fc_functionality1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality1.bias, 0)
        nn.init.normal_(self.fc_aesthetics1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics1.bias, 0)
        
        nn.init.normal_(self.fc_style2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_style2.bias, 0)
        nn.init.normal_(self.fc_semantics2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_semantics2.bias, 0)
        nn.init.normal_(self.fc_functionality2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_functionality2.bias, 0)
        nn.init.normal_(self.fc_aesthetics2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_aesthetics2.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        #for gconv in self.gconvs:
        #    verts = F.relu(gconv(verts, edges))
        for i in range(len(self.dims)-1):
            verts = getattr(self, 'gconv{}'.format(i))(verts, edges)
            verts = F.relu(getattr(self, 'norm{}'.format(i))(verts))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts_packed = packed_to_list(verts, verts_size)
        #verts_padded = list_to_padded(verts_packed)
        #out  = torch.mean(verts_padded, 1)
        verts_meaned = [torch.mean(xmesh.unsqueeze(0), dim=1) for xmesh in verts_packed]
        out = torch.cat(verts_meaned, dim=0)
        
        out = F.relu(self.fc1(out))
        out_style = self.fc_style2(F.relu(self.fc_style1(out)))
        out_semantics = self.fc_semantics2(F.relu(self.fc_semantics1(out)))
        out_functionality = self.fc_functionality2(F.relu(self.fc_functionality1(out)))
        out_aesthetics = self.fc_aesthetics2(F.relu(self.fc_aesthetics1(out)))
        #out = self.fc2(out)   
        # return out
        return [out_style, out_semantics, out_functionality, out_aesthetics]
    
