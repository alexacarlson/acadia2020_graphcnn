import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
import os, sys, pickle
import numpy as np
from pytorch3d.utils import ico_sphere
try:
    from .ico_objects import ico_disk
except:
    from scripts.ico_objects import ico_disk
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

sys.path.append('../scripts')
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
from scripts.obj2gif import render_mesh, images2gif, display_and_save_gif_ncluster_grid 

from gcnna.MeshCNN.models.layers.mesh import Mesh
from gcnna.MeshCNN.util.util import pad
from gcnna.MeshCNN.models.layers.mesh_prepare import extract_features, remove_non_manifolds, build_gemm

from mpl_toolkits.mplot3d import Axes3D
#import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

from pytorch3d.io import load_objs_as_meshes
import torch.nn.functional as F

#from PyGEL3D import gel
#from PyGEL3D import js
import pdb
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

device = torch.device("cuda")

import warnings
warnings.filterwarnings("ignore")

cos = torch.nn.CosineSimilarity(dim=0)


def plot_mesh(mesh=None, verts=None, faces=None):
    if mesh != None:
        save_obj('mesh.obj', mesh.verts_packed(), mesh.faces_packed())
    else:
        save_obj('mesh.obj', verts, faces)
    js.set_export_mode()
    m = gel.obj_load('mesh.obj')
    js.display(m, smooth=False)

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
    plt.show()

def gif_pointcloud(mesh, path=""):
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
        ax.view_init(i,i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 180), interval=100)
    anim.save(os.path.join(path, 'model.gif'), dpi=80, writer = matplotlib.animation.PillowWriter())

class FeatureVisualization():
    def __init__(self, model, src_mesh_name, exp, result_dir=os.getcwd()):
        """
        model: Trained Model
        src_mesh: sphere or disk
        exp: name of hte experiment
        """
        self.model = model
        self.model.net.eval()
        self.src_mesh_name = src_mesh_name
        self.exp = exp
        self.result_dir = os.path.join(result_dir, self.exp, 'viz_files')
        # self.result_dir = os.path.join('results/gcnna_data/', self.exp)
        os.makedirs(self.result_dir, exist_ok=True)
        
    def load_mesh(self, path):
        verts, faces, aux = load_obj(path)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        return mesh.cuda()

    def save_mesh(self, mesh, path):
        save_obj(path, mesh.verts_packed(), mesh.faces_packed())

    def plot_mesh_meshcnn_format(self, mesh):
        plot_mesh(verts=mesh.vs, faces=mesh.faces)

    def plot_mesh_meshcnn_format(self, meshverts, meshfaces):
        plot_mesh(verts=meshverts, faces=meshfaces)
        
    def normalize_verts(self, verts):
        # X
        if (verts[:,0].max() - verts[:,0].min()) != 0:
            verts[:,0] = ((verts[:,0] - verts[:,0].min())/(verts[:,0].max() - verts[:,0].min())) - 0.5
        else:
            verts[:,0] = 0.1

        # Y
        if (verts[:,1].max() - verts[:,1].min()) != 0:
            verts[:,1] = ((verts[:,1] - verts[:,1].min())/(verts[:,1].max() - verts[:,1].min())) - 0.5
        else:
            verts[:,1] = 0.1

        # Z
        if (verts[:,2].max() - verts[:,2].min()) != 0:
            verts[:,2] = ((verts[:,2] - verts[:,2].min())/(verts[:,2].max() - verts[:,2].min())) - 0.5
        else:
            verts[:,2] = 0.1
        return verts

    def PCA_svd(self, X, k=3, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n,1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
        H = (torch.eye(n) - h).cuda()
        X_center =  torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center) 
    #     print(u.t().shape, v.t().shape)
        components  = v[:k].t()
        explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components
                

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm


    def euclidian_loss(self, src_feats, trg_feats):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
        """
        distance_matrix = trg_feats.view(-1) - src_feats.view(-1)
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(src_feats, 2)
        return normalized_euclidian_distance
    
    def gram_matrix(self, feats):
        '''
        feats: N * F
        returns: F * F matrix 
        '''
        N, _ = feats.shape
        return torch.mm(feats.T, feats)/N
    
    def get_gram_loss(self, src_feats, trg_feats):
        src_gram = self.gram_matrix(src_feats)
        trg_gram = self.gram_matrix(trg_feats)
#         print(src_gram.shape, trg_gram.shape)
        return torch.sum((src_gram - trg_gram)**2)
    
    def rmse(self, src_feats, trg_feats):
        return torch.sqrt(torch.mean((src_feats - trg_feats)**2))

    def load_mesh_as_meshcnn_format_partnet(self, path, opt):
        mesh = load_objs_as_meshes([path],  load_textures=False)
        ## MESHES
        mesh_features_packed  = mesh.verts_packed().unsqueeze(0)
        ## normalize to unit sphere
        mesh_features_packed = mesh_features_packed - mesh_features_packed.mean(dim=1)
        mesh_features_packed/=max(mesh_features_packed.abs().max(dim=1)[0])
        mesh_edges_packed  = mesh.edges_packed().unsqueeze(0)
        meta = {'mesh': mesh}#, 'label': label}
        meta['features'] = mesh_features_packed.cuda() #mesh_features_padded
        meta['edges'] = mesh_edges_packed.cuda() #mesh_edges_padded
        meta['numverts'] = mesh_features_packed.shape[1]
        meta['faces'] = mesh.faces_packed().unsqueeze(0)
        return meta
    
    def load_mesh_as_meshcnn_format_partnet_BAD(self, path, opt):
        mesh = load_objs_as_meshes([path],  load_textures=False)
        ## MESHES
        max_num_verts = 4070
        mesh_features_packed  = mesh.verts_packed().unsqueeze(0)
        num_verts = mesh_features_packed.shape[1]
        numvstoadd = max_num_verts-mesh_features_packed.shape[1]
        if numvstoadd>0:
            mesh_features_padded = F.pad(input=mesh_features_packed, pad=(0, 0, 0, numvstoadd, 0, 0), mode='constant', value=0)
        else:
            mesh_features_padded = mesh_features_packed
        ## EDGES
        mesh_edges_packed  = mesh.edges_packed().unsqueeze(0)
        num_edges = mesh_edges_packed.shape[1]
        max_edge_num = 19012
        numedgestoadd = max_edge_num-mesh_edges_packed.shape[1]
        if numedgestoadd>0:
            mesh_edges_padded = F.pad(input=mesh_edges_packed, pad=(0, 0, 0, numedgestoadd, 0, 0), mode='constant', value=0)
        else:
            mesh_edges_padded = mesh_edges_packed·
        #
        ## Mean subtract
        #mean_std_cache = os.path.join(opt.dataroot, 'mean_std_cache.p')
        #with open(mean_std_cache, 'rb') as f:
        #    transform_dict = pickle.load(f)
        #    mean = transform_dict['mean']
        #    std = transform_dict['std']
        ##mesh_features_padded = ((mesh_features_padded - mean) / std).unsqueeze(0).float().cuda()
        #mesh_features_padded = ((mesh_features_padded - mean) / std).float()
        #
        meta = {'mesh': mesh} #, 'label': label}
        meta['numverts'] = num_verts
        meta['numedges'] = num_edges
        meta['features'] = mesh_features_padded.cuda()
        meta['edges'] = mesh_edges_padded.cuda()
        meta['faces'] = mesh.faces_packed().unsqueeze(0).cuda()
        return meta
    
    def load_mesh_as_meshcnn_format(self, path, opt):
        '''
        loads the data in Mesh format used in MeshCNN given the object file location
        '''
        mesh = Mesh(file=path, opt=opt, hold_history=False, export_folder=opt.export_folder)
        if opt.method == 'edge_cnn':
            mesh.features = pad(mesh.features, opt.ninput_edges)
            mean_std_cache = os.path.join(opt.dataroot, 'mean_std_cache.p')
            with open(mean_std_cache, 'rb') as f:
                transform_dict = pickle.load(f)
                mean = transform_dict['mean']
                std = transform_dict['std']
            mesh.features = ((mesh.features - mean) / std).unsqueeze(0).float().cuda()
        else:
            # print('Number of verts before padding: ', mesh.features.shape)
            mesh.features = pad(mesh.features, 252, dim=0, method='gcn_cnn')
            mesh.features = mesh.features.unsqueeze(0).cuda()
            mesh.edges = mesh.edges.unsqueeze(0).cuda()
            if opt.method =='zgcn_cnn':
                mesh.adj = mesh.adj.unsqueeze(0).cuda()

        return mesh

    def load_src_mesh_as_meshcnn_format(self, opt, pth):
        '''
        Loads source meshes in the Mesh format used in MeshCNN
        '''
        # if mesh_name == "sphere":
        #     src_mesh = ico_sphere(ico_level)
        # elif mesh_name =='disk':
        #     src_mesh = ico_disk(ico_level)
        # else:
        #     raise Exception('Invalid mesh name chosen')
        # save_obj('src_mesh.obj',src_mesh.verts_packed(), src_mesh.faces_packed())
        return self.load_mesh_as_meshcnn_format(pth, opt)

        
        
################### Feature Inversion ######################
class Feature_Inversion(FeatureVisualization):
    
    '''
    Class for feature inversion
    
    '''
    def __init__(self, model, src_mesh_name, exp, opt):
        super().__init__(model, src_mesh_name, exp) 
        self.opt = opt
        # self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt)

    def deform_mesh(self, mesh_data, opt):
        '''
        Function for intialize/re-initialize mesh with deformed vertices
        '''
        if opt.method == 'edge_cnn':
            mesh_data.edges = mesh_data.adj =  mesh_data.gemm_edges = mesh_data.sides = mesh_data.edges_count = mesh_data.ve = mesh_data.v_mask = mesh_data.edge_lengths = None
            mesh_data.edge_areas = []
            mesh_data.v_mask = torch.ones(len(mesh_data.vs), dtype=bool)
            mesh_data.faces, face_areas = remove_non_manifolds(mesh_data, mesh_data.faces)
            build_gemm(mesh_data, mesh_data.faces.detach().numpy(), face_areas.detach().numpy())
            mesh_data.features = extract_features(mesh_data)
            mesh_data.features = pad(mesh_data.features, opt.ninput_edges, mode='constant')
            mean_std_cache = os.path.join(opt.dataroot, 'mean_std_cache.p')
            with open(mean_std_cache, 'rb') as f:
                transform_dict = pickle.load(f)
                mean = torch.Tensor(transform_dict['mean'])
                std = torch.Tensor(transform_dict['std'])
            mesh_data.features = ((mesh_data.features - mean) / std).unsqueeze(0).float().cuda()
        return mesh_data

    def get_feats(self, mesh, layer):
        if self.opt.method == 'edge_cnn':
            return self.model.net.extract_feats(x=mesh.features, mesh=[mesh], layer=layer, verbose=False)
        elif self.opt.method == 'gcn_cnn':
            #return self.model.net.extract_feats(x=mesh.features, mesh=mesh.edges, layer=layer, verbose=False)
            return self.model.net.extract_feats(x=mesh['features'], mesh=mesh['edges'], layer=layer, verbose=False)
        elif self.opt.method == 'zgcn_cnn':
            return self.model.net.extract_feats(x=mesh.features, mesh=mesh.adj, layer=layer, verbose=False)

    def get_patch_data(self, patch_idx):
        idx = [patch_idx]
        for e in self.trg_mesh.edges[0]:
            if patch_idx in e:
                idx.append(e[e!=patch_idx])
        
        def create_src_patch(n):
            '''
            n = number of vertices
            '''
            x, y, z = .1,.1,.1
            n -= 1
            vs = [[x, y, z]]
            for i in range(n):
                vs.append([x+.4*np.cos((i*2*np.pi)/n), y+.4*np.sin((i*2*np.pi)/n), z])
            faces = [[0, i, i+1] for i in range(1, n)]
            faces.append([0, n, 1])
            verts, faces = torch.tensor(vs, dtype=torch.float32), torch.tensor(faces, dtype=torch.int)
            save_obj('disk.obj', verts, faces)

        create_src_patch(len(idx)) 
        src_mesh =  self.load_mesh_as_meshcnn_format('disk.obj', self.opt)

        return src_mesh, idx


    def invert_feats(self, trg_path, layer=None,filter=None, inv_method='sphere', patch_idx = 0,
                    lr=1, weights=None, iters = 200, verbose=False, normalize_cd=True):
        '''
        trg_path: target object path
        layer: layer to use for feature inversion
        filter: choose a group or a single neuron for feature inversion
        inv_method: choosing an intial mesh - sphere, patch, disk, random(to check the effect of geometry)
        patch_idx: if method is patch then choose a patch in the target mesh
        lr: learning rate for mesh deformation
        weights: weights for losses
        iters: number of iteration for mesh deformation
        verbose:
        normalize_cd: normalize chamfer distance so that it remains in the same range as other loss
        '''
        
        # Load target mesh and extract feats for the layer
        #self.trg_mesh = self.load_mesh_as_meshcnn_format(trg_path, self.opt)
        self.trg_mesh = self.load_mesh_as_meshcnn_format_partnet(trg_path, self.opt)
        
        trg_feats = self.get_feats(self.trg_mesh, layer)

        # Load model
        if inv_method == 'random_feat':
            trg_feats = torch.rand(trg_feats.size()).cuda()
        elif inv_method == 'patches':
            self.src_mesh, idx = self.get_patch_data(patch_idx)
            trg_feats = trg_feats[:, idx, :]
        elif inv_method == 'sphere':
            #self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'sphere.obj')
            self.src_mesh = self.load_mesh_as_meshcnn_format_partnet('sphere.obj',self.opt)
        elif inv_method == 'disk':
            #self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'disk.obj')
            self.src_mesh = self.load_mesh_as_meshcnn_format_partnet('disk.obj',self.opt)
        else:
            #self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, inv_method)
            self.src_mesh = self.load_mesh_as_meshcnn_format_partnet(inv_method,self.opt)

        ## optimize defrom_verts
        #if self.opt.method == 'edge_cnn':
        #    self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
        #    optimizer = torch.optim.Adam([self.src_mesh.vs], lr = lr)
        #else:
        #    self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
        #    optimizer = torch.optim.Adam([self.src_mesh.features], lr = lr)
        self.src_mesh['features'] = Variable(self.src_mesh['features'], requires_grad = True)
        optimizer = torch.optim.Adam([self.src_mesh['features']], lr = lr)

        max_val = 1
        # Run iterations
        for i in (range(iters)):
            optimizer.zero_grad()

            self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
            
            # Get the output from the model after a forward pass until target_layer for the source object
            src_feats = self.get_feats(self.src_mesh, layer)
            #if self.opt.method == 'edge_cnn':
            #    src_mesh_p3d = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces]).cuda()
            #else:
            #    src_mesh_p3d = Meshes(verts=[self.src_mesh.features.squeeze(0)], faces=[self.src_mesh.faces]).cuda()
            src_mesh_p3d = Meshes(verts=[self.src_mesh['features'].squeeze(0)], faces=[self.src_mesh['faces'].squeeze(0)]).cuda()

            # Losses
            # cosine_loss = cos(trg_feats.detach(), src_feats)
            # euc_loss = .1*self.euclidian_loss(trg_feats.detach(), src_feats)
            # rmse_loss = self.rmse(trg_feats.detach(), src_feats)

            # latent_loss = 0.000005*(torch.mean(torch.abs(trg_feats.detach() - src_feats)))
            # gram_loss = 1e-17*self.get_gram_loss(src_feats, trg_feats.detach())

            if 'res' in layer:
                if filter is None:
                    feat_loss, _ = chamfer_distance(trg_feats.detach(), src_feats)
                        # print(feat_loss.detach(), max_val)
                else: 
                    try:
                        if len(filter) > 1:
                            feat_loss, _ = chamfer_distance(trg_feats[:,:,filter].detach(), src_feats[:,:,filter])
                    except:
                        feat_loss, _ = chamfer_distance(trg_feats[:,:,filter].unsqueeze(-1).detach(), src_feats[:,:,filter].unsqueeze(-1))
            else:
                feat_loss = self.euclidian_loss(trg_feats.detach(), src_feats)

            fl = feat_loss.clone().detach()
            if i == 0 and fl >= 1 and normalize_cd:
                max_val = str(int(fl))
                max_val = 10**int(len(max_val))

            # pdb.set_trace()
            
            feat_loss = (weights['feat_loss'] * feat_loss)/max_val
            
            # Regularizations
            laplacian_loss = weights['lap_loss']*mesh_laplacian_smoothing(src_mesh_p3d, method="uniform")
            edge_loss = weights['edge_loss']*mesh_edge_loss(src_mesh_p3d)
            # normal_loss = 1*mesh_normal_consistency(self.new_src_mesh)
            
            # Sum all to optimize
            loss =  feat_loss + laplacian_loss + edge_loss
            
            # Step
            loss.backward()
            optimizer.step()
            
            # Generate image every 5 iterations
            if i % int(iters/5) == 0 and verbose:
                print('Iteration:', str(i), 
                      'Loss:', loss.item(), 
                      'Feats Loss', feat_loss.item(),
                    #   "Gram Loss:", gram_loss.item(),
                    #   "Cosine Loss:", cosine_loss.item(),
                    #   "Latent Loss:", latent_loss.item(),
                    #   "Euc Loss:", euc_loss.item(), 
                      "Lap Loss:", laplacian_loss.item(), 
                      "Edge Loss:", edge_loss.item(),
                    #  ,"Normal Loss:", normal_loss.item(),
                     )
                # if self.opt.method != 'edge_cnn':
                #     self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()
                # new_src_mesh = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces])
                # plot_pointcloud(new_src_mesh)
                

            # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/10
        if self.opt.method != 'edge_cnn':
            #self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()
            self.src_mesh['features'] = self.src_mesh['features'].detach().squeeze(0).cpu()
        
        # Calculate Final Chamfer Distance
        #new_src_mesh = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces])
        new_src_mesh = Meshes(verts=[self.src_mesh['features'].squeeze(0)], faces=[self.src_mesh['faces'].squeeze(0)])
        # trg_mesh = Meshes(verts=[self.trg_mesh.vs], faces=[self.trg_mesh.faces])

        # sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        # sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # out_cd, _ = chamfer_distance(sample_trg, sample_src)
        out_cd = None
        return out_cd, new_src_mesh
        
        # os.makedirs(self.result_dir, exist_ok=True)
        # if filter == None:
        #     obj_path = os.path.join(self.result_dir, str(float("{:.4f}".format(out_cd))) + '_' + self.src_mesh_name+'_'+layer+'.obj')
        # else:
        #     obj_path = os.path.join(self.result_dir, self.src_mesh_name + '_' + layer + '_' + str(filter) + '.obj') 
        # save_obj(obj_path, new_src_mesh.verts_packed(), new_src_mesh.faces_packed())


    def invert_feats_multilayers(self, trg_path, layers=None,filter=None, inv_method='sphere', patch_idx = 0,
                    lr=1, weights=None, iters = 200, verbose=False, normalize_cd=True):
        '''
        trg_path: target object path
        layer: layer to use for feature inversion
        filter: choose a group or a single neuron for feature inversion
        inv_method: choosing an intial mesh - sphere, patch, disk, random(to check the effect of geometry)
        patch_idx: if method is patch then choose a patch in the target mesh
        lr: learning rate for mesh deformation
        weights: weights for losses
        iters: number of iteration for mesh deformation
        verbose:
        normalize_cd: normalize chamfer distance so that it remains in the same range as other loss
        '''
        
        # Load target mesh and extract feats for the layer
        self.trg_mesh = self.load_mesh_as_meshcnn_format(trg_path, self.opt)
        
        #trg_feats = self.get_feats(self.trg_mesh, layer)
        trg_feats_list = []
        for layer in layers:
            trg_feats = self.get_feats(self.trg_mesh, layer)
            trg_feats_list.append(trg_feats)
            
        # Load model
        if inv_method == 'random_feat':
            trg_feats = torch.rand(trg_feats.size()).cuda()
        elif inv_method == 'patches':
            self.src_mesh, idx = self.get_patch_data(patch_idx)
            trg_feats = trg_feats[:, idx, :]
        elif inv_method == 'sphere':
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'sphere.obj')
        elif inv_method == 'disk':
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'disk.obj')
        else:
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, inv_method)

        # optimize defrom_verts
        if self.opt.method == 'edge_cnn':
            self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
            optimizer = torch.optim.Adam([self.src_mesh.vs], lr = lr)
        else:
            self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
            optimizer = torch.optim.Adam([self.src_mesh.features], lr = lr)
        

        max_val = 1
        # Run iterations
        for i in (range(iters)):
            optimizer.zero_grad()

            self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
            
            ## Get the output from the model after a forward pass until target_layer for the source object
            #src_feats = self.get_feats(self.src_mesh, layer)
            src_feats_list = []
            for layer in layers:
                src_feats = self.get_feats(self.src_mesh, layer)
                src_feats_list.append(src_feats)
            
            if self.opt.method == 'edge_cnn':
                src_mesh_p3d = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces]).cuda()
            else:
                src_mesh_p3d = Meshes(verts=[self.src_mesh.features.squeeze(0)], faces=[self.src_mesh.faces]).cuda()

            # Losses
            # cosine_loss = cos(trg_feats.detach(), src_feats)
            # euc_loss = .1*self.euclidian_loss(trg_feats.detach(), src_feats)
            # rmse_loss = self.rmse(trg_feats.detach(), src_feats)

            # latent_loss = 0.000005*(torch.mean(torch.abs(trg_feats.detach() - src_feats)))
            # gram_loss = 1e-17*self.get_gram_loss(src_feats, trg_feats.detach())

            if 'res' in layer:
                if filter is None:
                    #feat_loss, _ = chamfer_distance(trg_feats.detach(), src_feats)
                    feat_loss=0
                    for trg_feats in trg_feats_list:
                        feat_loss_, _ = chamfer_distance(trg_feats.detach(), src_feats)
                        feat_loss+=feat_loss_
                    # print(feat_loss.detach(), max_val)
                else: 
                    try:
                        if len(filter) > 1:
                            #feat_loss, _ = chamfer_distance(trg_feats[:,:,filter].detach(), src_feats[:,:,filter])
                            feat_loss=0
                            for trg_feats, src_feats in zip(trg_feats_list, src_feats_list):
                                feat_loss_, _ = chamfer_distance(trg_feats[:,:,filter].detach(), src_feats[:,:,filter])
                                feat_loss+=feat_loss_
                    except:
                        #feat_loss, _ = chamfer_distance(trg_feats[:,:,filter].unsqueeze(-1).detach(), src_feats[:,:,filter].unsqueeze(-1))
                        feat_loss=0
                        for trg_feats in trg_feats_list:
                            feat_loss, _ = chamfer_distance(trg_feats[:,:,filter].unsqueeze(-1).detach(), src_feats[:,:,filter].unsqueeze(-1))
                            feat_loss+=feat_loss_
            else:
                #feat_loss = self.euclidian_loss(trg_feats.detach(), src_feats)
                feat_loss=0
                for trg_feats, src_feats in zip(trg_feats_list, src_feats_list):
                    feat_loss = self.euclidian_loss(trg_feats.detach(), src_feats)
                    feat_loss+=feat_loss_
                        
            fl = feat_loss.clone().detach()
            if i == 0 and fl >= 1 and normalize_cd:
                max_val = str(int(fl))
                max_val = 10**int(len(max_val))

            # pdb.set_trace()
            
            feat_loss = (weights['feat_loss'] * feat_loss)/max_val
            
            # Regularizations
            laplacian_loss = weights['lap_loss']*mesh_laplacian_smoothing(src_mesh_p3d, method="uniform")
            edge_loss = weights['edge_loss']*mesh_edge_loss(src_mesh_p3d)
            # normal_loss = 1*mesh_normal_consistency(self.new_src_mesh)
            
            # Sum all to optimize
            loss =  feat_loss + laplacian_loss + edge_loss
            
            # Step
            loss.backward()
            optimizer.step()
            
            # Generate image every 5 iterations
            if i % int(iters/5) == 0 and verbose:
                print('Iteration:', str(i), 
                      'Loss:', loss.item(), 
                      'Feats Loss', feat_loss.item(),
                    #   "Gram Loss:", gram_loss.item(),
                    #   "Cosine Loss:", cosine_loss.item(),
                    #   "Latent Loss:", latent_loss.item(),
                    #   "Euc Loss:", euc_loss.item(), 
                      "Lap Loss:", laplacian_loss.item(), 
                      "Edge Loss:", edge_loss.item(),
                    #  ,"Normal Loss:", normal_loss.item(),
                     )
                # if self.opt.method != 'edge_cnn':
                #     self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()
                # new_src_mesh = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces])
                # plot_pointcloud(new_src_mesh)
                

            # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/10
        if self.opt.method != 'edge_cnn':
            self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()
        
        # Calculate Final Chamfer Distance
        new_src_mesh = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces])
        # trg_mesh = Meshes(verts=[self.trg_mesh.vs], faces=[self.trg_mesh.faces])

        # sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        # sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # out_cd, _ = chamfer_distance(sample_trg, sample_src)
        out_cd = None
        return out_cd, new_src_mesh
        
        # os.makedirs(self.result_dir, exist_ok=True)
        # if filter == None:
        #     obj_path = os.path.join(self.result_dir, str(float("{:.4f}".format(out_cd))) + '_' + self.src_mesh_name+'_'+layer+'.obj')
        # else:
        #     obj_path = os.path.join(self.result_dir, self.src_mesh_name + '_' + layer + '_' + str(filter) + '.obj') 
        # save_obj(obj_path, new_src_mesh.verts_packed(), new_src_mesh.faces_packed())
        
    def dream_layer(self, trg_path, layer=None,filter=None, inv_method='sphere',
                    lr=1, weights=None, iters = 200, verbose=False, normalize_cd=True):
        # Load intial mesh
        self.trg_mesh = self.load_mesh_as_meshcnn_format(trg_path, self.opt)
        
        trg_feats = self.get_feats(self.trg_mesh, layer)

        # Load model
        if inv_method == 'random_feat':
            trg_feats = torch.rand(trg_feats.size()).cuda()
        elif inv_method == 'patches':
            self.src_mesh, idx = self.get_patch_data(patch_idx)
            trg_feats = trg_feats[:, idx, :]
        elif inv_method == 'sphere':
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'sphere.obj')
        elif inv_method == 'disk':
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, 'disk.obj')
        else:
            self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt, inv_method)

        # optimize defrom_verts
        if self.opt.method == 'edge_cnn':
            self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
            optimizer = torch.optim.Adam([self.src_mesh.vs], lr = lr)
        else:
            self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
            optimizer = torch.optim.SGD([self.src_mesh.features], lr = lr)
        

        max_val = 1
        # Run iterations
        for i in (range(iters)):
            optimizer.zero_grad()

            self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
            
            # Get the output from the model after a forward pass until target_layer for the source object
            src_feats = self.get_feats(self.src_mesh, layer).squeeze()

            if self.opt.method == 'edge_cnn':
                src_mesh_p3d = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces]).cuda()
            else:
                src_mesh_p3d = Meshes(verts=[self.src_mesh.features.squeeze(0)], faces=[self.src_mesh.faces]).cuda()

            # print(src_feats.shape)
            feat_loss = -torch.sum(src_feats[:,filter])
            
            # fl = feat_loss.clone().detach()
            # if i == 0 and fl >= 1 and normalize_cd:
            #     max_val = str(int(fl))
            #     max_val = 10**int(len(max_val))

            # pdb.set_trace()
            
            feat_loss = (weights['feat_loss'] * feat_loss)/max_val
            
            # Regularizations
            laplacian_loss = weights['lap_loss']*mesh_laplacian_smoothing(src_mesh_p3d, method="uniform")
            edge_loss = weights['edge_loss']*mesh_edge_loss(src_mesh_p3d)
            # normal_loss = 1*mesh_normal_consistency(self.new_src_mesh)
            
            # Sum all to optimize
            loss =  feat_loss + laplacian_loss + edge_loss
            
            # Step
            loss.backward()
            optimizer.step()
            
            # Generate image every 5 iterations
            if i % int(iters/5) == 0 and verbose:
                print('Iteration:', str(i), 
                      'Loss:', loss.item(), 
                      'Feats Loss', feat_loss.item(),
                    #   "Gram Loss:", gram_loss.item(),
                    #   "Cosine Loss:", cosine_loss.item(),
                    #   "Latent Loss:", latent_loss.item(),
                    #   "Euc Loss:", euc_loss.item(), 
                      "Lap Loss:", laplacian_loss.item(), 
                      "Edge Loss:", edge_loss.item(),
                    #  ,"Normal Loss:", normal_loss.item(),
                     )
                # if self.opt.method != 'edge_cnn':
                #     self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()
                # new_src_mesh = Meshes(verts=[self.src_mesh.vs], faces=[self.src_mesh.faces])
                # plot_pointcloud(new_src_mesh)
                

            # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/10
        if self.opt.method != 'edge_cnn':
            self.src_mesh.vs = self.src_mesh.features.detach().squeeze(0).cpu()

            # # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/5



# ################### Saliency Maps ######################
class SaliencyViz(FeatureVisualization):
    def __init__(self, model, exp, opt, result_dir=os.getcwd()):
        self.model = model
        self.exp = exp 
        self.opt = opt
        self.result_dir = os.path.join(result_dir, self.exp, 'viz_files')
        # self.result_dir = os.path.join('results/gcnna_data/', self.exp)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # self.src_mesh = self.load_src_mesh_as_meshcnn_format(self.opt)

    def deform_mesh(self, mesh_data, opt):
        
        if opt.method == 'edge_cnn':
            mesh_data.edges = mesh_data.adj =  mesh_data.gemm_edges = mesh_data.sides = mesh_data.edges_count = mesh_data.ve = mesh_data.v_mask = mesh_data.edge_lengths = None
            mesh_data.edge_areas = []
            mesh_data.v_mask = torch.ones(len(mesh_data.vs), dtype=bool)
            mesh_data.faces, face_areas = remove_non_manifolds(mesh_data, mesh_data.faces)
            build_gemm(mesh_data, mesh_data.faces.detach().numpy(), face_areas.detach().numpy())
            mesh_data.features = extract_features(mesh_data)
            mesh_data.features = pad(mesh_data.features, opt.ninput_edges, mode='constant')
            mean_std_cache = os.path.join(opt.dataroot, 'mean_std_cache.p')
            with open(mean_std_cache, 'rb') as f:
                transform_dict = pickle.load(f)
                mean = torch.Tensor(transform_dict['mean'])
                std = torch.Tensor(transform_dict['std'])
            mesh_data.features = ((mesh_data.features - mean) / std).unsqueeze(0).float().cuda()
        return mesh_data

    
    def get_feats(self, mesh, layer):
        if self.opt.method == 'edge_cnn':
            return self.model.net.extract_feats(x=mesh.features, mesh=[mesh], layer=layer, verbose=False)
        elif self.opt.method == 'gcn_cnn':
            #return self.model.net.extract_feats(x=mesh.features, mesh=mesh.edges, layer=layer, verbose=False)
            #print(mesh['features'].shape, mesh['edges'].shape)
            return self.model.net.extract_feats(x=mesh['features'], mesh=mesh['edges'], layer=layer, verbose=False)
        elif self.opt.method == 'zgcn_cnn':
            return self.model.net.extract_feats(x=mesh.features, mesh=mesh.adj, layer=layer, verbose=False)

    def get_colors(self, inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        #pdb.set_trace()
        return colormap(norm(inp))

    def get_saliency(self, mesh_path, layer=None, filter=0, return_intensity = False, components = 4):
        '''
        mesh_path: target object path
        layer: layer to use for feature inversion
        filter: choose a group or a single neuron for saliency map
        return_intensity: return the intensity values from saliency maps
        components: number of components for agglomerative clustering
        '''
        if not os.path.exists(os.path.join(self.result_dir,'%s_%sclusterexp'%(layer, str(components)))):
            os.mkdir(os.path.join(self.result_dir,'%s_%sclusterexp'%(layer, str(components))))
        # Load target mesh and extract feats for the layer
        self.src_mesh = self.load_mesh_as_meshcnn_format_partnet(mesh_path, self.opt)
        #print(self.src_mesh['features'])
        #print(self.src_mesh['edges'])
        #pdb.set_trace()
        if self.opt.method == 'edge_cnn':
            self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
        else:
            #self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
            self.src_mesh['features'] = Variable(self.src_mesh['features'], requires_grad = True)
        intensity = []

        # update the mesh with gradients
        self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
        
        # Get the output from the model after a forward pass until target_layer for the source object
        src_feats = self.get_feats(self.src_mesh, layer).squeeze()

        # F , V  (64, 252)
        #print(src_feats.shape)
        src_feats = src_feats.permute(1,0)
        self.src_f = src_feats

        ## Cluster activations 
        cls = AgglomerativeClustering(n_clusters=components, linkage='ward')
        k = cls.fit_predict(src_feats.detach().cpu())

        for cluster in tqdm(range(components)):

            # Losses for each cluster
            loss = torch.sum(src_feats[k==cluster, :])
            #loss = torch.sum(src_feats[cluster, :])
            loss.backward(retain_graph=True)

            # # calculate the intendity values
            if self.opt.method != 'edge_cnn':      
                #nverts_intensity = torch.norm(self.src_mesh.features.grad.squeeze(), dim=1).cpu().numpy() 
                nverts_intensity = torch.norm(self.src_mesh['features'].grad.squeeze(), dim=1).cpu().numpy() 
            else:
                #nverts_intensity = torch.norm(self.src_mesh.vs.grad, dim=1).cpu().numpy()
                nverts_intensity = torch.norm(self.src_mesh['features'].grad, dim=1).cpu().numpy()
            ## normalize
            nverts_intensity = (nverts_intensity - nverts_intensity.min())/(nverts_intensity.max() - nverts_intensity.min())
            #print(np.unique(nverts_intensity, axis=0))
            # pdb.set_trace()
            # if return_intensity:
            #     return nverts_intensity
            intensity.append(nverts_intensity)
        
            # dump files and visualize
            filepath = os.path.join(self.result_dir, '%s_%sclusterexp'%(layer, str(components)), 'gcn_model_%s_'%self.opt.modelfname+'_'.join([layer, str(cluster)]) )
            
            num_verts = len(nverts_intensity)
            #assert num_verts == 252
            #
            ## Color map for intensity
            nv = self.src_mesh['numverts']
            rmesh = self.src_mesh['features'].squeeze()
            rfaces = self.src_mesh['faces'].squeeze()
            verts_rgb_ = 255.*torch.Tensor(self.get_colors(nverts_intensity, plt.cm.coolwarm)[:,:3]).unsqueeze(0).to(device)
            #textured_mesh = Meshes(verts=[self.src_mesh.vs.to(device)], faces=[self.src_mesh.faces.to(device)], textures=Textures(verts_rgb=verts_rgb_colors))
            t_mesh = Meshes(verts=[rmesh.to(device)], faces=[rfaces.to(device)], 
                            textures=TexturesVertex(verts_features=verts_rgb_) )
            all_images = render_mesh(t_mesh, elevation = 45, dist_=5, batch_size=50,  imageSize=256)
            all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
            #pdb.set_trace()
            images2gif(all_images_, filepath)

            # reset grads
            if self.opt.method == 'edge_cnn':
                #self.src_mesh.vs.grad = None
                self.src_mesh['features'].grad = None
            else:
                self.src_mesh['features'].grad = None

        return k, intensity
    
    def get_saliency_multipleclusters_exp(self, mesh_path, layer=None, filter=0, return_intensity = False, max_components = 4):
        '''
        mesh_path: target object path
        layer: layer to use for feature inversion
        filter: choose a group or a single neuron for saliency map
        return_intensity: return the intensity values from saliency maps
        components: number of components for agglomerative clustering
        '''
        if not os.path.exists(os.path.join(self.result_dir,'%s_multiclusterexp'%layer)):
            os.mkdir(os.path.join(self.result_dir,'%s_multiclusterexp'%layer))
            
        # Load target mesh and extract feats for the layer
        #self.src_mesh = self.load_mesh_as_meshcnn_format(mesh_path, self.opt)
        self.src_mesh = self.load_mesh_as_meshcnn_format_partnet(mesh_path, self.opt)

        #if self.opt.method == 'edge_cnn':
        #    self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
        #else:
        #    self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
        self.src_mesh['features'] = Variable(self.src_mesh['features'], requires_grad = True)
        print(self.src_mesh['features'].shape)
        #intensity = []
        #
        ## update the mesh with gradients
        self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
        #
        ## Get the output from the model after a forward pass until target_layer for the source object
        src_feats = self.get_feats(self.src_mesh, layer).squeeze()
        #
        ## F , V  (64, 252)
        #print(src_feats.shape)
        src_feats = src_feats.permute(1,0)
        #
        clusters_dict={}
        intensity_dict={}
        #fig, axs = plt.subplots(max_components, max_components, figsize=(3*max_components, 3*max_components))
        for components in np.arange(1,max_components+1):
            print('Processing for %s clusters'%str(components))
            intensity = []
            # Cluster activations 
            cls = AgglomerativeClustering(n_clusters=components, linkage='ward')
            k = cls.fit_predict(src_feats.detach().cpu())
            #
            percluster_img_dict={}
            percluster_intensity_dict={}
            #for cluster in tqdm(range(components)):
            for cluster in range(components):
                # Losses for each cluster
                loss = torch.sum(src_feats[k==cluster, :])
                loss.backward(retain_graph=True)

                ## calculate the intendity values
                if self.opt.method != 'edge_cnn':      
                    #nverts_intensity = torch.norm(self.src_mesh.features.grad.squeeze(), dim=1).cpu().numpy() 
                    nverts_intensity = torch.norm(self.src_mesh['features'].grad, dim=2).cpu().numpy()
                else:
                    #nverts_intensity = torch.norm(self.src_mesh.vs.grad, dim=1).cpu().numpy()
                    nverts_intensity = torch.norm(self.src_mesh['features'].grad, dim=2).cpu().numpy()
                #print(self.src_mesh['features'].grad.shape)
                #print(nverts_intensity.shape)
                ## normalize
                nverts_intensity = (nverts_intensity- nverts_intensity.min())/(nverts_intensity.max() - nverts_intensity.min())
                # pdb.set_trace()
                # if return_intensity:
                #     return nverts_intensity
                #print(nverts_intensity.shape)
                intensity.append(nverts_intensity)
        
                ## dump files and visualize
                filepath = os.path.join(self.result_dir, '%s_multiclusterexp'%layer, '_'.join([layer, str(cluster)]) )
                #
                num_verts = len(nverts_intensity)
                #
                #assert num_verts == 252
                ## Color map for intensity
                #verts_rgb_colors = 255*torch.Tensor(self.get_colors(nverts_intensity,plt.cm.coolwarm)[:,:3]).unsqueeze(0).to(device)
                #textured_mesh = Meshes(verts=[self.src_mesh.vs.to(device)], 
                #                   faces=[self.src_mesh.faces.to(device)], 
                #                   textures=Textures(verts_rgb=verts_rgb_colors))
                #                    #
                #all_images = render_mesh(textured_mesh, elevation = 315, dist_=2, batch_size=50, device=device, imageSize=512)
                #all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
                ## Color map for intensity
                #print(self.src_mesh['features'].squeeze().shape)
                nv = self.src_mesh['numverts']
                rmesh = self.src_mesh['features'] #.squeeze()
                rfaces = self.src_mesh['faces'] #.squeeze()
                verts_rgb_ = 255.*torch.Tensor(self.get_colors(nverts_intensity, plt.cm.coolwarm)[:,:,:3]).to(device)
                #print(rmesh.shape)
                #print(verts_rgb_.shape)
                t_mesh = Meshes(verts=rmesh.to(device), faces=rfaces.to(device),            textures=TexturesVertex(verts_features=verts_rgb_) )
                all_images = render_mesh(t_mesh, elevation = 45, dist_=5, batch_size=50, device=device, imageSize=256)
                all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
                #display_and_save_gif(all_images_, filepath)
                percluster_img_dict[' '.join([layer,'clustr%sof%s'%(str(cluster), str(components-1))])] = all_images_ 
                percluster_intensity_dict[' '.join([layer,'clustr%sof%s'%(str(cluster), str(components-1))])] =nverts_intensity
                #
                # reset grads
                if self.opt.method == 'edge_cnn':
                    #self.src_mesh.vs.grad = None
                    self.src_mesh['features'].grad = None
                else:
                    self.src_mesh['features'].grad = None     
            clusters_dict[str(components)]=k
            intensity_dict[str(components)]=percluster_intensity_dict
            display_and_save_gif_ncluster_grid(percluster_img_dict, os.path.join(self.result_dir,'%s_multiclusterexp'%layer))
            ## add suplots
            #for j,img_key in enumerate(percluster_img_dict):
            #    axs[components-1,j].imshow(percluster_img_dict[img_key][1])
            #    axs[components-1,j].set_title(img_key.replace(' ','\n'), {'fontsize':10})
        #[axs[i,j].axis('off') for i in range(max_components) for j in range(max_components)]
        #plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        #plt.show()
        #plt.savefig(os.path.join(self.result_dir,'%s_multiclusterexp'%layer,'gcn_model_'+self.opt.modelfname+'.png'))
        return clusters_dict, intensity_dict

    def get_saliency_corrparts_exp(self, mesh_path, label_path, layer=None, filter=0, return_intensity = False, components = 4):
        '''
        mesh_path: target object path
        layer: layer to use for feature inversion
        filter: choose a group or a single neuron for saliency map
        return_intensity: return the intensity values from saliency maps
        components: number of components for agglomerative clustering
        '''
        if not os.path.exists(os.path.join(self.result_dir,'%s_%sclusterexp'%(layer, str(components)))):
            os.mkdir(os.path.join(self.result_dir,'%s_%sclusterexp'%(layer, str(components))))
            
        # Load target mesh and extract feats for the layer
        self.src_mesh = self.load_mesh_as_meshcnn_format_partnet(mesh_path, self.opt)
        
        # Load label
        def load_label(fn):
            with open(fn, 'r') as fin:
                lines = [item.rstrip() for item in fin]
                label = np.array([int(line) for line in lines], dtype=np.int32)
            return label
        self. src_label = load_label(label_path)
        pdb.set_trace()
        
        if self.opt.method == 'edge_cnn':
            self.src_mesh.vs = Variable(self.src_mesh.vs, requires_grad = True)
        else:
            #self.src_mesh.features = Variable(self.src_mesh.features, requires_grad = True)
            self.src_mesh['features'] = Variable(self.src_mesh['features'], requires_grad = True)
        intensity = []

        # update the mesh with gradients
        self.src_mesh = self.deform_mesh(self.src_mesh, self.opt)    
        
        # Get the output from the model after a forward pass until target_layer for the source object
        src_feats = self.get_feats(self.src_mesh, layer).squeeze()

        # F , V  (64, 252)
        #print(src_feats.shape)
        src_feats = src_feats.permute(1,0)
        self.src_f = src_feats

        ## Cluster activations 
        cls = AgglomerativeClustering(n_clusters=components, linkage='ward')
        k = cls.fit_predict(src_feats.detach().cpu())

        for cluster in tqdm(range(components)):

            # Losses for each cluster
            loss = torch.sum(src_feats[k==cluster, :])
            #loss = torch.sum(src_feats[cluster, :])
            loss.backward(retain_graph=True)

            # # calculate the intendity values
            if self.opt.method != 'edge_cnn':      
                #nverts_intensity = torch.norm(self.src_mesh.features.grad.squeeze(), dim=1).cpu().numpy() 
                nverts_intensity = torch.norm(self.src_mesh['features'].grad.squeeze(), dim=1).cpu().numpy() 
            else:
                #nverts_intensity = torch.norm(self.src_mesh.vs.grad, dim=1).cpu().numpy()
                nverts_intensity = torch.norm(self.src_mesh['features'].grad, dim=1).cpu().numpy()
            ## normalize
            nverts_intensity = (nverts_intensity - nverts_intensity.min())/(nverts_intensity.max() - nverts_intensity.min())
            #print(np.unique(nverts_intensity, axis=0))
            # pdb.set_trace()
            # if return_intensity:
            #     return nverts_intensity
            intensity.append(nverts_intensity)
        
            # dump files and visualize
            filepath = os.path.join(self.result_dir, '%s_%sclusterexp'%(layer, str(components)), 'gcn_model_%s_'%self.opt.modelfname+'_'.join([layer, str(cluster)]) )
            
            num_verts = len(nverts_intensity)
            #assert num_verts == 252
            #
            ## Color map for intensity
            nv = self.src_mesh['numverts']
            rmesh = self.src_mesh['features'].squeeze()
            rfaces = self.src_mesh['faces'].squeeze()
            verts_rgb_ = 255.*torch.Tensor(self.get_colors(nverts_intensity, plt.cm.coolwarm)[:,:3]).unsqueeze(0).to(device)
            #textured_mesh = Meshes(verts=[self.src_mesh.vs.to(device)], faces=[self.src_mesh.faces.to(device)], textures=Textures(verts_rgb=verts_rgb_colors))
            t_mesh = Meshes(verts=[rmesh.to(device)], faces=[rfaces.to(device)], 
                            textures=TexturesVertex(verts_features=verts_rgb_) )
            all_images = render_mesh(t_mesh, elevation = 45, dist_=5, batch_size=50, device=device, imageSize=256)
            all_images_ = [Image.fromarray(np.uint8(img.detach().cpu().squeeze().numpy())) for img in all_images]
            #pdb.set_trace()
            images2gif(all_images_, filepath)

            # reset grads
            if self.opt.method == 'edge_cnn':
                #self.src_mesh.vs.grad = None
                self.src_mesh['features'].grad = None
            else:
                self.src_mesh['features'].grad = None

        return k, intensity
    





