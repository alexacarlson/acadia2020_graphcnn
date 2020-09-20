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



def deep_dream_loss(input_mesh, classification_model, layer_name='fc2', node=None):
        # compute loss
        #nnfeatures =  classification_model.get_forward_fcfeats(input_mesh, layer_name)
        nnfeatures =  classification_model.get_forward_feats(input_mesh, layer_name)
        nnfeatures=torch.squeeze(nnfeatures)
        if 'fc' in layer_name:
            loss = -torch.sum(nnfeatures[node]**2)/nnfeatures[node].numel()
        elif 'gconv' in layer_name:
            vert = node[0]
            feati = node[1]
            loss = -torch.sum(nnfeatures[vert, feati]**2)/nnfeatures[vert, feati].numel()
        #loss = -torch.sum(nnfeatures[:,node]**2)/nnfeatures[:,node].numel()
        #var_line_length = get_var_line_length_loss(self.mesh.vertices, self.mesh.faces)
        #loss += self.lambda_length * var_line_length
        return loss


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

def gif_pointcloud(mesh, title=""):
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
    anim.save(title+'.gif', dpi=80, writer = matplotlib.animation.PillowWriter())
    #anim.save(title+'.gif', dpi=80, writer='imagemagick')
    #ax.view_init(190, 30)
    #plt.savefig(title+'.png')
    #plt.show()

if __name__ == "__main__":
    ## settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str)
    parser.add_argument('-cfg', '--config_path', type=str)
    parser.add_argument('-alr', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-wm', '--which_starting_mesh', type=str, default='sphere')
    parser.add_argument('-wv', '--which_vertex', type=int, default=None)
    parser.add_argument('-wf', '--which_feature', type=int, default=100)
    parser.add_argument('-wl', '--which_layer', type=str, default='fc1')
    parser.add_argument('-lap', '--mesh_laplacian_smoothing', type=bool, default=True)
    parser.add_argument('-ni', '--num_iteration', type=int, default=300)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.5)
    parser.add_argument('-ib', '--init_bias', type=str, default='(0,0,0)')
    #parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()  

    ## Set the device
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    #
    ## make the output directory if necessary 
    output_dir = '_'.join(['/root/results_dreaming', str(args.which_layer), str(args.which_vertex), str(args.which_feature)])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ## load in trained graph convolutional NN classification model
    graphconv_model_path = '/root/graphconv_classification_results/model/exp_02_17_02_06_41_overfit'
    #content_mesh_path = args.filename_obj
    idx_best_loss=33
    #LAYER_ID = 1
    #NODE_ID = 100
    ITERS = args.num_iteration  
    MESH_LR =1.0

    cfg = Config(args.config_path)
    #
    ## SET UP seed mesh for dreaming
    if args.which_starting_mesh=='sphere':
        ## FOR loading in sphere; initialize the source shape to be a sphere of radius 1
        src_mesh = ico_sphere(4, device)
    elif args.which_starting_mesh=='plane':
        ## FOR loading in plane
        src_mesh = ico_plane(2., 3., 2, precision = 1.0, z = 0.0, color = None, device=device)
    elif os.path.isfile(args.which_starting_mesh):
        ## FOR loading in input mesh from file
        verts, faces, aux=load_obj(args.which_starting_mesh)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        src_mesh = Meshes(verts=[verts], faces=[faces_idx])
    else:
        print('Please specify a valid input mesh, one of: sphere, plane, or filepath to obj mesh file')
        sys.exit()

    ## SET UP model and optimizer
    classification_net_model = GraphConvClf(cfg).cuda()
    classification_net_model.load_state_dict(torch.load(graphconv_model_path+'/model@epoch'+str(idx_best_loss)+'.pkl', map_location=torch.device('cpu'))['state_dict'])
    ## freeze the parameters for the classification model
    for pp in classification_net_model.parameters():
        pp.requires_grad=False
        #
    ## set up optimization; we will learn to deform the source mesh by offsetting its vertices. The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = optim.Adam([deform_verts], lr=MESH_LR,)
    #optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # Plot period for the losses
    plot_period = 50
    # --------------------------------------------------------------------------------------------
    #   DREAMING LOOP
    # --------------------------------------------------------------------------------------------
    print('\n ***************** Deep Dream *****************')  
    for iter_ in tqdm(range(ITERS)):    
        #
        ## zero the parameter gradients
        optimizer.zero_grad()
        #
        ## Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        #print(deform_verts)
        #
        ## Calculate loss on deformed mesh
        #loss = deep_dream_loss(new_src_mesh, classification_net_model, 'fc2', 0)
        if 'fc' in args.which_layer:
            loss = deep_dream_loss(new_src_mesh, classification_net_model, args.which_layer, args.which_feature)
        elif 'gconv' in args.which_layer:
            if not args.which_vertex:
                print("To dream using a graph convolutional layer, you need to specify both a vertex and a feature using --which_vertex and --which_feature")
                sys.exit()
            loss = deep_dream_loss(new_src_mesh, classification_net_model, args.which_layer, [args.which_vertex, args.which_feature])
        if args.mesh_laplacian_smoothing: 
            ## add mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            loss+= 0.1*loss_laplacian
            #
        ## Plot mesh
        if iter_ % plot_period == 0:
            if torch.sum(torch.isnan(loss)).item()>0:
                print('nan values in loss:', torch.sum(torch.isnan(loss)).item())
            if torch.sum(torch.isnan(deform_verts)).item()>0:
                print('nan values in deform verts:', torch.sum(torch.isnan(deform_verts)).item())
            gif_pointcloud(new_src_mesh, title=os.path.join(output_dir, os.path.splitext(args.output_filename)[0]+"iter_%d" % iter_))
            #plot_pointcloud(new_src_mesh, title=args.filename_output+"iter_%d" % iter_)
            print('Iteration: '+str(iter_) + ' Loss: '+str(loss))
            #
        ## apply loss 
        loss.backward()
        optimizer.step()
    ## final obj shape
    gif_pointcloud(new_src_mesh, title=os.path.join(output_dir, os.path.splitext(args.output_filename)[0]+"iter_%d" % iter_))
    ## Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)    
    ## save output mesh
    save_obj(os.path.splitext(args.output_filename)[0], final_verts, final_faces)

