import numpy as np
#import pandas as pd
import torch, os
from torch.utils.data import Dataset
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import packed_to_list
from style_transfer.config import Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from style_transfer.config import Config
from style_transfer.utils.torch_utils import train_val_split_mesh2aesthetics, save_checkpoint, accuracy
from style_transfer.data.datasets import ShapenetDataset, mesh2aesthetics_Dataset

print("getting config")
_C = Config("/tf/notebooks/acadia2020_graphcnn_src/config/mesh2aesthetics_train.yml")
print("getting dataset list")
trn_objs_list, val_objs_list = train_val_split_mesh2aesthetics(config=_C)
trn_dataset = mesh2aesthetics_Dataset(_C, trn_objs_list)
print("getting dataset loader")
collate_fn = mesh2aesthetics_Dataset.collate_fn 
trn_dataloader = DataLoader(trn_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            collate_fn=collate_fn, 
                            num_workers=8)
print("getting dataset split")
trn_objs_list, val_objs_list = train_val_split_mesh2aesthetics(config=_C)
collate_fn = mesh2aesthetics_Dataset.collate_fn

shape_check = []
counter = 0
for i, data in enumerate(trn_dataloader, 0):
    label = data[0].cuda()
    mesh = data[1].cuda()
    meshname = data[2]
    #print(mesh.verts_packed().shape)
    shape_check.append(mesh.verts_packed().shape[0])
    if mesh.verts_packed().shape[0]>50000:
        counter+=1
        print(meshname[0].split('/home/alexandracarlson/Desktop/acadia2020_3daesthetics_dataset/Separated/')[1], mesh.verts_packed().shape[0])

print(i)
print(min(shape_check))
print(max(shape_check))
print(counter)