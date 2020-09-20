import numpy as np
import pandas as pd
import torch, os
from torch.utils.data import Dataset
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import packed_to_list
from style_transfer.config import Config
from tqdm import tqdm

class ShapenetDataset(Dataset):
    def __init__(self, cfg, obj_list):
#         self.device = cfg.DEVICE
        #self.transform = cfg.SHAPENET_DATA.TRANSFORM
        self.obj_list = obj_list
        
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cls, obj_name = self.obj_list[idx]
        verts, faces, aux = load_obj(obj_name)
        return cls, verts, faces.verts_idx

    def collate_fn(batch):
        cls, verts, faces = zip(*batch)
        
        if verts[0] is not None and faces[0] is not None:
            meshes = Meshes(verts=list(verts), faces=list(faces))
        else:
            meshes = None
        
#         ### VERTS ###
#         verts = meshes.verts_packed()
#         verts_idx = meshes.verts_packed_to_mesh_idx()
#         verts_size = tuple(verts_idx.unique(return_counts=True)[1])
#         verts = packed_to_list(verts, split_size=verts_size)

#         ### EDGES ###
#         edges = meshes.edges_packed()
#         edges_idx = meshes.edges_packed_to_mesh_idx()
#         edge_size = tuple(edges_idx.unique(return_counts=True)[1])
#         edges = packed_to_list(edges, split_size=edge_size)
        
        ### Convert to Tensor
        cls = torch.Tensor(cls).to(dtype=int)
        return cls, meshes

class mesh2acoustic_Dataset(Dataset):
    def __init__(self, cfg, obj_list):
#         self.device = cfg.DEVICE
        #self.transform = cfg.SHAPENET_DATA.TRANSFORM
        self.obj_list = obj_list
        
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #cls, obj_name = self.obj_list[idx]
        obj_params, obj_name = self.obj_list[idx]
        verts, faces, aux = load_obj(obj_name)
        return obj_params, verts, faces.verts_idx

    def collate_fn(batch):
        #cls, verts, faces = zip(*batch)
        obj_params, verts, faces = zip(*batch)
        
        if verts[0] is not None and faces[0] is not None:
            meshes = Meshes(verts=list(verts), faces=list(faces))
        else:
            meshes = None
        
#         ### VERTS ###
#         verts = meshes.verts_packed()
#         verts_idx = meshes.verts_packed_to_mesh_idx()
#         verts_size = tuple(verts_idx.unique(return_counts=True)[1])
#         verts = packed_to_list(verts, split_size=verts_size)

#         ### EDGES ###
#         edges = meshes.edges_packed()
#         edges_idx = meshes.edges_packed_to_mesh_idx()
#         edge_size = tuple(edges_idx.unique(return_counts=True)[1])
#         edges = packed_to_list(edges, split_size=edge_size)
        
        ### Convert to Tensor
        cls = torch.Tensor(obj_params)#.to(dtype=int)
        return cls, meshes