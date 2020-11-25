import argparse
import logging
import os,sys
from typing import Type
import random 
from tqdm import tqdm
import pdb
import torch
import pickle
import pandas as pd
import numpy as np
import os
import yaml
import re
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
import cv2
from style_transfer.config import Config
from style_transfer.models.base_nn import GraphConvClf
from style_transfer.data.datasets import ShapenetDataset
from style_transfer.config import Config
from style_transfer.utils.torch_utils import train_val_split, save_checkpoint, accuracy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
from ico_plane import ico_plane
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
import warnings
warnings.filterwarnings("ignore")

from PIL import Image·
#from pytorch3d.structures import Textures·
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
from obj2gif import render_mesh, images2gif, display_and_save_gif_ncluster_grid

from mpl_toolkits.mplot3d import Axes3D
#import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.savefig(title+'.png')
    #plt.show()

def gif_pointcloud(mesh, filename, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    def update(i):
        ax.view_init(190,i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
    # Set up formatting for the movie files
    #writer1 = matplotlib.animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800) #matplotlib.animation.writers['ffmpeg']
    #writer1 = Writer1(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, dpi=80, writer = matplotlib.animation.PillowWriter())
    #anim.save(title+'.gif', dpi=80, writer='imagemagick')
    #ax.view_init(190, 30)
    #plt.savefig(title+'.png')
    #plt.show()
    
def param_loss(input_mesh, net_model, desired_params):
        # compute loss
        nnpred =  torch.squeeze(net_model.forward(input_mesh))
        #var_line_length = get_var_line_length_loss(self.mesh.vertices, self.mesh.faces)
        #loss += self.lambda_length * var_line_length
        #return nn.MSELoss([nnpred], [desired_params])
        #return torch.sum((nnpred - desired_params) ** 2)
        return loss
        
if __name__ == "__main__":
    ## settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str)
    parser.add_argument('-cfg', '--config_path', type=str)
    parser.add_argument('-alr', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    #parser.add_argument('-sif', '--silhouette_img_ref', type=str, default=None)
    parser.add_argument('-gnet', '--trained_graphnet_weights', type=str, default='/storage/training_results/exp_03_10_11_57_39_c')
    parser.add_argument('-smesh', '--starting_mesh', type=str, default='sphere')
    parser.add_argument('-stp', '--style_param', type=str, default='baroque')
    parser.add_argument('-semp', '--semantic_param', type=str, default='house')
    parser.add_argument('-funcp', '--functionvalue_param', type=str, default=5)
    parser.add_argument('-sesthp', '--aestheticvalue_param', type=str, default=5)
    parser.add_argument('-maw', '--mesh_param_optim_weights', type=float, default=1.)
    #parser.add_argument('-msw', '--mesh_multisilhouette_optim_weight', type=float, default=1.)
    #parser.add_argument('-cpf', '--camera_positions_file', type=str, default=None)
    parser.add_argument('-lap', '--mesh_laplacian_smoothing', type=lambda x:bool(util.strtobool(x)), default=True)
    #parser.add_argument('-ap', '--mesh_acousticparam_optim', type=lambda x:bool(util.strtobool(x)), default=True)
    #parser.add_argument('-so', '--mesh_multisilhouette_optim', type=lambda x:bool(util.strtobool(x)), default=True)
    parser.add_argument('-ni', '--num_iteration', type=int, default=501)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.5)
    parser.add_argument('-ib', '--init_bias', type=str, default='(0,0,0)')
    #parser.add_argument('-g', '--gpu', type=int, default=0)
    args_ = parser.parse_args()  

    ## Set the device
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    #
    ## make the output directory if necessary 
    meshid = os.path.splitext(os.path.split(args_.starting_mesh)[1])[0]
    output_dir = '/storage/results_%s_optim_style%s_semantic%s_func%s_aesth%s'%(str(meshid), str(args_.style_param), str(args_.semantic_param), str(args_.functionvalue_param), str(args_.aestheticvalue_param))
    print('Saving optim results to %s'%(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    def get_colors(self, inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        #pdb.set_trace()
        return colormap(norm(inp))
    
    ## ---- SET UP seed mesh for class visual saliency ---- ##
    if args_.starting_mesh=='sphere':
        ## FOR loading in sphere; initialize the source shape to be a sphere of radius 1
        src_mesh = ico_sphere(4, device)
    elif args_.starting_mesh=='plane':
        ## FOR loading in plane
        src_mesh = ico_plane(2., 3., 2, precision = 1.0, z = 0.0, color = None, device=device)
    elif os.path.isfile(args_.starting_mesh):
        ## FOR loading in input mesh from file
        verts, faces, aux=load_obj(args_.starting_mesh)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        src_mesh = Meshes(verts=[verts], faces=[faces_idx])
    else:
        print('Please specify a valid input mesh, one of: sphere, plane, or filepath to obj mesh file')
        sys.exit()


    ## ---- SET UP/load in trained graph convolutional NN classification model ---- ##
    graphconv_model_path = args_.trained_graphnet_weights 
    
    cfg = Config(args_.config_path)
    
    #desired_params = torch.tensor([float(ap) for ap in args_.which_params.split(',')], dtype=torch.float32, device=device)
    STYLECLASSESDICT={'baroque':0, 'modern':0, 'moden':0, 'classic':0, '(Insert Label)':0, 'cubist':1, 'cubims':1, 'cu':1, 'cubism':1, 'Cubism':1}
    
    SEMANTICCLASSESDICT={'house':0, 'House':0, 'column':1, 'Column':1}
    
    stylep = STYLECLASSESDICT[args_.style_param]
    semp = SEMANTICCLASSESDICT[args_.semantic_param]
    funcp = int(args_.functionvalue_param)-1 if int(args_.functionvalue_param) <=4 else 4-1
    aesthp= int(args_.aestheticvalue_param)-1 if int(args_.aestheticvalue_param)<=5 else 5-1
    
    #desired_params  = torch.Tensor([stylep, semp, funcp, aesthp]).long().cuda()
    
    desired_params  =[torch.Tensor([stylep]).cuda().long(), torch.Tensor([semp]).cuda().long(), torch.Tensor([funcp]).cuda().long(), torch.Tensor([aesthp]).cuda().long()]
    #print(desired_params)
    
    ## ---- SET UP model and optimizer ---- ##
    net_model = GraphConvClf(cfg).cuda()
    net_model.load_state_dict(torch.load(graphconv_model_path, map_location=torch.device('cpu'))['state_dict'])
    ## freeze the parameters for the classification model
    for pp in net_model.parameters():
        pp.requires_grad=False
            
    ## ---- SET UP optimization variables of mesh ---- ##
    ITERS = args_.num_iteration  
    MESH_LR =0.05 #1.0
    #
    ## set up optimization; 
    ## we will learn to deform the source mesh by offsetting its vertices. 
    ## The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = optim.Adam([deform_verts], lr=MESH_LR,)

    # Plot period for the losses
    plot_period = 50

    # --------------------------------------------------------------------------------------------
    #   DEFORMATION LOOP
    # --------------------------------------------------------------------------------------------
    print('\n ***************** Deforming *****************') 
    criterion_style = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_sem = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_func = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_aesth = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    
    for iter_ in range(ITERS):    
        #
        loss=0
        ## zero the parameter gradients
        optimizer.zero_grad()
        #
        ## Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        #print(deform_verts)
        
        ## Calculate loss on deformed mesh
        outputs = net_model.forward(new_src_mesh)
        
        nverts_intensity = torch.norm(self.src_mesh['features'].grad, dim=1).cpu().numpy()
        nverts_intensity = (nverts_intensity - nverts_intensity.min())/(nverts_intensity.max() - nverts_intensity.min())
        verts_rgb_ = 255.*torch.Tensor(get_colors(nverts_intensity, plt.cm.coolwarm)[:,:3]).unsqueeze(0).to(device)
        t_mesh = Meshes(verts=[rmesh.to(device)],faces=[rfaces.to(device)],textures=TexturesVertex(verts_features=verts_rgb_) )
        all_images = render_mesh(t_mesh, elevation = 45, dist_=5, batch_size=50, device=device, imageSize=256)
        all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
        images2gif(all_images_, filepath) 
        
        #print(outputs[0].shape, desired_params[0].shape, desired_params[0])
        #print(outputs[1].shape, desired_params[1].shape, desired_params[1])
        #print(outputs[2].shape, desired_params[2].shape, desired_params[2])
        #print(outputs[3].shape, desired_params[3].shape, desired_params[3])
        loss_style = criterion_style(outputs[0], desired_params[0])
        loss_semantic = criterion_sem(outputs[1], desired_params[1])
        loss_functionality = criterion_func(outputs[2], desired_params[2])
        loss_aesthetic = criterion_aesth(outputs[3], desired_params[3])
        loss_ = loss_style + loss_semantic + loss_functionality + loss_aesthetic
        #loss_ = param_loss(new_src_mesh, optim_net_model, desired_params, args_.mesh_aestheticparam_optim_weights)
        loss+=loss_

        if args_.mesh_laplacian_smoothing: 
            ## add mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            loss+=0.5*loss_laplacian

        ## Plot mesh
        if iter_ % plot_period == 0:
            ## make sure there are no nans in the mesh
            if torch.sum(torch.isnan(loss)).item()>0:
                print('nan values in loss:', torch.sum(torch.isnan(loss)).item())
            if torch.sum(torch.isnan(deform_verts)).item()>0:
                print('nan values in deform verts:', torch.sum(torch.isnan(deform_verts)).item())
                #
            ## plot point cloud render of deformed mesh
            ofname=os.path.join(output_dir, os.path.splitext(args_.output_filename)[0]+"iter_%d" % iter_)
            gif_pointcloud(new_src_mesh, os.path.join(output_dir, ofname+'.gif'), title=ofname)
            #plot_pointcloud(new_src_mesh, title=args_.filename_output+"iter_%d" % iter_)
            #
            print('Iteration: '+str(iter_) + ' Loss: '+str(loss.cpu().detach().numpy()))
            #
        ## apply loss 
        loss.backward()
        optimizer.step()
        ##
    ## final obj shape
    ofname=os.path.join(output_dir, os.path.splitext(args_.output_filename)[0]+"_final")
    gif_pointcloud(new_src_mesh, os.path.join(output_dir, ofname+".gif"), title=ofname)
    ## Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)    
    ## save output mesh
    save_obj(os.path.join(output_dir, args_.output_filename), final_verts, final_faces)
