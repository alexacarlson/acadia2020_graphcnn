import argparse
import logging
import os,sys
from typing import Type
import random 
from tqdm import tqdm
import pdb
import torch
import pickle
import numpy as np
import os
import yaml
import re
from torch import nn, optim
from torch.utils.data import DataLoader
#from pytorch3d.structures import Textures 
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
# from pytorch3d.io import load_objs_as_meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from mpl_toolkits.mplot3d import Axes3D

import matplotlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
from PIL import Image 
import numpy as np

# Data structures and functions for mesh rendering
from pytorch3d.structures import Meshes
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
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader, HardFlatShader
)
#from pytorch3d.renderer.mesh.texturing import interpolate_texture_map, interpolate_vertex_colors
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)

from IPython.display import display, Image, HTML, Markdown
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
import warnings
warnings.filterwarnings("ignore")

#class SimpleShader(nn.Module):
#    def __init__(self, device="cpu"):
#        super().__init__()
#    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
#        #pdb.set_trace()
#        pixel_colors = interpolate_vertex_colors(fragments, meshes)
#        images = hard_rgb_blend(pixel_colors, fragments)
#        return images # (N, H, W, 3) RGBA image

def render_mesh(mesh, elevation, dist_, batch_size, device, imageSize):
    # Initialize an OpenGL perspective camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    #R, T = look_at_view_transform(150.0, 1.0, 180) 
    #dd = camera_loc[0]
    #el = camera_loc[1]
    #az = camera_loc[2]
    #batch_size=50
    #
    meshes = mesh.extend(batch_size)
    #
    #el = torch.linspace(0, 180, batch_size)
    az = torch.linspace(0, 180, batch_size)
    R, T = look_at_view_transform(dist=dist_, elev=elevation, azim=az) 
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T) 
    #
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=imageSize, 
        blur_radius=0.0, 
        faces_per_pixel=1
    )   
    #
    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -5.0]])#, [0.0, 0.0, 5.0], [0.0, -5.0, 0.0], [0.0, 5.0, 0.0]])    

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings),
        shader=HardPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights 
    ))
    ## render images
    images = renderer(meshes)
    return images


def images2gif(image_list, filepath):
    image_list[0].save(filepath+'.gif', save_all=True, append_images=image_list, optimize=False, duration=400, loop=0)
    
def display_and_save_gif(image_list, filepath):
    gifloc = filepath+'.gif'
    image_list[0].save(gifloc, save_all=True, append_images=image_list, optimize=False, duration=400, loop=0)
    #Image(filename=filepath+'.gif') 
    #from IPython.display import HTML
    #HTML('<img src=%s+".gif">'%filepath)
    #from IPython.display import display, Image
    display(Image(filename=gifloc))
    #return filepath+'.gif'

#import ipyplot
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#def display_and_save_gif_ncluster_grid(percluster_img_dict, folderpath, row_height='auto'):
#    giflocs = []
#    giftitles = []
#    ##########
#    for img_key in percluster_img_dict:
#        image_list = percluster_img_dict[img_key]
#        gifloc = os.path.join(folderpath,img_key)+'.gif'
#        image_list[0].save(gifloc, save_all=True, append_images=image_list, optimize=False, duration=400, loop=0)
#        giflocs.append(gifloc)
#        imagetitle=img_key.replace(' ','\n') #' '.join(img_key.split('_'))
#        giftitles.append(imagetitle)
#    #ipyplot.plot_images(giflocs, giftitles, img_width=150)
#    imgs = [mpimg.imread(name) for name in giflocs]
#    n_rows = 1
#    n_cols = len(imgs)
#    #print(len(giflocs))
#    #print(n_rows)
#    #print(n_cols)
#    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_cols))
#    if n_cols==1:
#        [axs.imshow(imgs[0]), axs.axis('off'), axs.set_title(giftitles[0])]
#    else:
#        #[(axs[i,j].imshow(imgs[i*n_cols + j]), axs[i][j].axis('off')) for i in range(n_rows) for j in range(n_cols)]
#        [(axs[j].imshow(imgs[j]), axs[j].axis('off'), axs[j].set_title(giftitles[j], {'fontsize':10})) for j in range(n_cols)]
#        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)#

def render_main(starting_mesh_path, camera_elevation, camera_rdistance, batch_size, image_size, output_filename, descr=''):
 
    ## Set the device
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    #
    verts, faces, aux=load_obj(starting_mesh_path)
    faces_idx = faces.verts_idx.to(device)
    #pdb.set_trace()
    gverts = verts.to(device)
    gverts.requires_grad=True
    src_mesh = Meshes(verts=[gverts], faces=[faces_idx])

    # print('\n ***************** Rendering Mesh as gif *****************')  
    ## render as mesh 
    num_verts = verts.shape[0]
    verts_rgb_colors = 128*torch.ones([1, num_verts, 3]).to(device)
    textured_mesh = Meshes(verts=[verts.to(device)], faces=[faces_idx.to(device)], textures=TexturesVertex(verts_features=verts_rgb_colors))
    #pdb.set_trace()
    all_images = render_mesh(textured_mesh, camera_elevation, camera_rdistance, batch_size, device, image_size)
    all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
    #pdb.set_trace()
    filepath = os.path.join(output_filename, descr+'_'+os.path.splitext(os.path.split(starting_mesh_path)[1])[0])
    images2gif(all_images_, filepath)

if __name__ == "__main__":
    ## settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str)
    parser.add_argument('-wm', '--which_starting_mesh', type=str, default='sphere')
    parser.add_argument('-bs', '--batch_size', type=float, default=30)
    parser.add_argument('-ims', '--image_size', type=int, default=512)
    parser.add_argument('-camD', '--camera_rdistance', type=float, default=10, help='Radial distance of camera from origin')
    parser.add_argument('-camEl', '--camera_elevation', type=float, default=45, help='degree Elevation of camera from origin')
    #parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args() 
    starting_mesh_path = args.which_starting_mesh
    camera_elevation = args.camera_elevation 
    camera_rdistance = args. camera_rdistance
    batch_size = args.batch_size
    image_size = args.image_size
    output_filename = args.output_filename
    render_main(starting_mesh_path, camera_elevation, camera_rdistance, batch_size, image_size, output_filename)

