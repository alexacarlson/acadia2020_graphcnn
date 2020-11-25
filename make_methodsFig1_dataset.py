import argparse
import logging
import os,sys
import numpy as np
from typing import Type
import random 
from tqdm import tqdm
import pdb
import torch
import pickle
import os
import yaml
import re

import matplotlib
#matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import mesh_laplacian_smoothing
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)
#import scipy.misc
from distutils import util
from style_transfer.config import Config
from style_transfer.config import Config

from ico_plane import ico_plane
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
#from pytorch3d.structures import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
#from obj2gif import render_mesh, images2gif
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

device="cuda:0"
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


# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(3.0, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -5.0]])

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

######### MAKE GRID FIGURE ############
counter = 0
which_view = 10
fs = (20,12)
dpi = 400
rr = 42
cc = 50
fig1, ax = plt.subplots(rr, cc,sharex=True, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=fs, dpi=dpi)
[ax[ii,jj].axis('off') for ii in range(rr) for jj in range(cc)]
rowCounter = 0
colCounter = 0
for i, data in enumerate(trn_dataloader, 0):
    print('rendering %s'%str(i))
    label = data[0].to(device)
    mesh = data[1].to(device)
    #
    #faces_idx = faces.verts_idx.to(device)
    nverts = mesh.verts_packed().shape[0]
    verts = mesh.verts_packed() 
    ## center to sphere at origin 0
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    verts_rgb_ = 0.5*torch.ones((nverts,3)).unsqueeze(0).to(device)
    t_mesh = Meshes(verts=[verts.to(device)], faces=[mesh.faces_packed().to(device)], textures=TexturesVertex(verts_features=verts_rgb_) )
    #all_images = render_mesh(t_mesh, elevation = 45, dist_=5, batch_size=1, device=device, imageSize=256)
    #all_images = render_mesh(t_mesh, elevation = 2, dist_=5, batch_size=5, device=device, imageSize=256)
    #all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
    images = renderer(t_mesh)
    #print(images.min())
    #print(images.max())
    #disp_img = all_images_[0]
    #disp_img.save(os.path.join(os.getcwd(), 'methods_fig1_test.png'))
    #print(np.min(disp_img))
    #print(np.max(disp_img))
    #print(np.unique(disp_img))
    ax[rowCounter,colCounter].imshow(images[0, ..., :3].cpu().numpy())
    #ax[rowCounter,colCounter].axis('off')
    colCounter+=1
    if colCounter==cc:
        rowCounter+=1
        colCounter=0

#plt.gcf()
fig1.savefig(os.path.join(os.getcwd(), 'methods_fig1_dataset.png'))
#plt.show()
