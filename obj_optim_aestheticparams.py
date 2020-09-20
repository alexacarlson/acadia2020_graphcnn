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


def mesh_multisilhouette_optim(input_mesh, camera_poses, silhouette_img_ref, silhouette_renderer, device):
    ## we want to render an image base loss (think like neural renderer vertex optim) but have it be from aorund the entire object
    #dist_=15.
    #el_ = 0.
    #camera_poses = [[dist_, el_, 0.],[dist_, el_, 90.],[dist_, el_, 180.],[dist_, el_, 270.]]
    loss=0
    silhouette_img_list=[]
    for dist, el, az in camera_poses:
        # Get the position of the camera based on the spherical angles
        R, T = look_at_view_transform(dist, el, az, device=device)
        # Render the input mesh providing the values of R and T. 
        silhouette_img = silhouette_renderer(meshes_world=input_mesh, R=R, T=T)
        # Calculate the silhouette loss for this angle
        loss += torch.sum((silhouette_img[...,3] - silhouette_img_ref) ** 2)
        silhouette_img_list.append(silhouette_img)
    return loss, silhouette_img_list
    #
    ## Get the position of the camera based on the spherical angles
    #R, T = look_at_view_transform(dist_, el_, 0., device=device)
    ## Render the input mesh providing the values of R and T. 
    #silhouette_img = silhouette_renderer(meshes_world=input_mesh, R=R, T=T)
    ## Calculate the silhouette loss for this angle
    ##pdb.set_trace()
    #loss = torch.sum((silhouette_img[...,3] - silhouette_img_ref) ** 2)
    #return loss, [silhouette_img]

def acousticparam_loss(input_mesh, net_model, desired_params):
        # compute loss
        nnpred =  torch.squeeze(net_model.forward(input_mesh))
        #var_line_length = get_var_line_length_loss(self.mesh.vertices, self.mesh.faces)
        #loss += self.lambda_length * var_line_length
        #return nn.MSELoss([nnpred], [desired_params])
        return torch.sum((nnpred - desired_params) ** 2)

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

def sublot_intermediate_renders(image_ref, img_list, outfile):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(image_ref.cpu().detach().numpy().squeeze())  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    for ii, simg in enumerate(img_list,2):
        silhouette = simg.cpu().detach().numpy()
        #pdb.set_trace()
        plt.subplot(1, 5, ii)
        plt.imshow(silhouette[...,3].squeeze())
        plt.grid(False)
    fig.savefig(outfile)

if __name__ == "__main__":
    ## settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str)
    parser.add_argument('-cfg', '--config_path', type=str)
    parser.add_argument('-alr', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-sif', '--silhouette_img_ref', type=str, default=None)
    parser.add_argument('-gnet', '--trained_graphnet_weights', type=str, default='/storage/mesh2acoustic_training_results/exp_03_10_11_57_39_c')
    parser.add_argument('-wm', '--which_starting_mesh', type=str, default='sphere')
    parser.add_argument('-wp', '--which_acoustic_params', type=str, default=None)
    parser.add_argument('-maw', '--mesh_acousticparam_optim_weight', type=float, default=1.)
    parser.add_argument('-msw', '--mesh_multisilhouette_optim_weight', type=float, default=1.)
    parser.add_argument('-cpf', '--camera_positions_file', type=str, default=None)
    parser.add_argument('-lap', '--mesh_laplacian_smoothing', type=lambda x:bool(util.strtobool(x)), default=True)
    parser.add_argument('-ap', '--mesh_acousticparam_optim', type=lambda x:bool(util.strtobool(x)), default=True)
    parser.add_argument('-so', '--mesh_multisilhouette_optim', type=lambda x:bool(util.strtobool(x)), default=True)
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
    if args_.mesh_multisilhouette_optim and args_.mesh_acousticparam_optim:
        output_dir = '_'.join(['/storage/results_acousticoptim', str(args_.which_acoustic_params), 'multicamVertoptim', os.path.splitext(os.path.split(args_.silhouette_img_ref)[1])[0]])
    elif args_.mesh_acousticparam_optim:
        output_dir = '_'.join(['/storage/results_acousticoptim', str(args_.which_acoustic_params)])
    elif args_.mesh_multisilhouette_optim:
        output_dir = '_'.join(['/storage/results_multicamVertoptim', os.path.splitext(os.path.split(args_.silhouette_img_ref)[1])[0]])
    else:
        print('need some optimization criterion, specify mesh_acousticparam_optim, mesh_multisilhouette_optim, or both')
    print('Saving optim results to %s'%(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ## ---- SET UP seed mesh for dreaming ---- ##
    if args_.which_starting_mesh=='sphere':
        ## FOR loading in sphere; initialize the source shape to be a sphere of radius 1
        src_mesh = ico_sphere(4, device)
    elif args_.which_starting_mesh=='plane':
        ## FOR loading in plane
        src_mesh = ico_plane(2., 3., 2, precision = 1.0, z = 0.0, color = None, device=device)
    elif os.path.isfile(args_.which_starting_mesh):
        ## FOR loading in input mesh from file
        verts, faces, aux=load_obj(args_.which_starting_mesh)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        src_mesh = Meshes(verts=[verts], faces=[faces_idx])
    else:
        print('Please specify a valid input mesh, one of: sphere, plane, or filepath to obj mesh file')
        sys.exit()

    ## ---- SET UP cameras for silhouette rendering ---- ##
    if args_.mesh_multisilhouette_optim:
        ## load in silhouette reference image
        silhouette_ref = (1./255.)*cv2.imread(args_.silhouette_img_ref)
        silhouette_ref = cv2.resize(silhouette_ref, (256, 256))
        silhouette_ref = torch.from_numpy((silhouette_ref[..., :3].max(-1) != 0).astype(np.float32)).to(device)
        #silhouette_ref = torch.tensor([silhouette_ref], dtype=torch.float32, device=device)
        #
        # Process camera positions
        if args_.camera_positions_file:
            with open(args_.camera_positions_file,'r') as ff:
                camera_lines = [x.strip() for x in ff.readlines()]
            camera_poses=[[float(cl.split(' ')[0]), float(cl.split(' ')[1]), float(cl.split(' ')[2])] for cl in camera_lines]
        else:
            dist_=15.
            el_=0.
            camera_poses=[[dist_, el_, 0.],[dist_, el_, 90.],[dist_, el_, 180.],[dist_, el_, 270.]]
        # Initialize an OpenGL perspective camera.
        cameras = OpenGLPerspectiveCameras(device=device)
        # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
        # edges. Refer to blending.py for more details. 
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 256x256. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py
        # for an explanation of this parameter. 
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params))

    ## Select the viewpoint using spherical angles  
    #distance = 10   # distance from camera to the object
    #elevation = 50.0   # angle of elevation in degrees
    #azimuth = 10.0  # No rotation so the camera is positioned on the +Z axis. 
    ## Get the position of the camera based on the spherical angles
    #R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    ## Render the teapot providing the values of R and T. 
    #silhouete = silhouette_renderer(meshes_world=src_mesh, R=R, T=T)
    #simg = silhouete.cpu().numpy()
    #cv2.imwrite('/root/silhouette_renderer_test.png', simg.squeeze())
    ##pdb.set_trace()

    ## ---- SET UP/load in trained graph convolutional NN classification model ---- ##
    if args_.mesh_acousticparam_optim:
        #graphconv_model_path = '/root/graphconvnet_classification_results/model/exp_02_17_02_06_41_overfit'
        graphconv_model_path = args_.trained_graphnet_weights #'/storage/graphconvnet_acousticparampred_results/exp_03_10_11_57_39_concerthalloptim'
        cfg = Config(args_.config_path)
        desired_acoustic_params = torch.tensor([float(ap) for ap in args_.which_acoustic_params.split(',')], dtype=torch.float32, device=device)
        ## ---- SET UP model and optimizer ---- ##
        acousticoptim_net_model = GraphConvClf(cfg).cuda()
        #acousticoptim_net_model.load_state_dict(torch.load(graphconv_model_path+'/model@epoch'+str(idx_best_loss)+'.pkl', map_location=torch.device('cpu'))['state_dict'])
        acousticoptim_net_model.load_state_dict(torch.load(graphconv_model_path, map_location=torch.device('cpu'))['state_dict'])
        ## freeze the parameters for the classification model
        for pp in acousticoptim_net_model.parameters():
            pp.requires_grad=False
            
    ## ---- SET UP optimization variables of mesh ---- ##
    ITERS = args_.num_iteration  
    MESH_LR =0.05 #1.0
    ## set up optimization; we will learn to deform the source mesh by offsetting its vertices. The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = optim.Adam([deform_verts], lr=MESH_LR,)
    #optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # Plot period for the losses
    plot_period = 50

    # --------------------------------------------------------------------------------------------
    #   DEFORMATION LOOP
    # --------------------------------------------------------------------------------------------
    print('\n ***************** Deforming *****************') 

    #for iter_ in tqdm(range(ITERS)):    
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
        if args_.mesh_acousticparam_optim:
            loss_acoustic = acousticparam_loss(new_src_mesh, acousticoptim_net_model, desired_acoustic_params)
            loss+=args_.mesh_acousticparam_optim_weight*loss_acoustic

        if args_.mesh_multisilhouette_optim:
            loss_sil, sil_images = mesh_multisilhouette_optim(new_src_mesh, camera_poses, silhouette_ref, silhouette_renderer, device)
            loss+=args_.mesh_multisilhouette_optim_weight*loss_sil

        if args_.mesh_laplacian_smoothing: 
            ## add mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            loss+=0.1*loss_laplacian

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
            ## if using a silhoutte loss, view silhouettes from each camera
            if args_.mesh_multisilhouette_optim:
                sublot_intermediate_renders(silhouette_ref, sil_images, os.path.join(output_dir, ofname+".png"))
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
