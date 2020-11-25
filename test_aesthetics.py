import argparse
import logging
import os,sys
#from typing import Type
import random 
#from tqdm import tqdm

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from style_transfer.data.datasets import ShapenetDataset, mesh2aesthetics_Dataset
from style_transfer.models.base_nn import GraphConvClf,GraphConvClf2,GraphConvClf3,GraphConvClf_singlesemclass,GraphConvClf_singletask
from style_transfer.config import Config
from style_transfer.utils.torch_utils import train_val_split_mesh2aesthetics, save_checkpoint, accuracy, load_network
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("Run testing for a particular trained model.")
parser.add_argument("--config-yml", required=True, help="Path to a config file for specified phase.")
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the results directory.",
)


logger: logging.Logger = logging.getLogger(__name__)

def test_model_loss(valmodel, val_dataloader, vloss_weightvars, _C):
    # -------------------------------------------------------------
    #   VALIDATION
    # -------------------------------------------------------------
    task_flag = _C.SHAPENET_DATA.WHICH_TASK=='all'
    vcriterion_style = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    vcriterion_sem = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    vcriterion_func = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    vcriterion_aesth = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss(      
    if _C.GCC.WHICH_GCN_FN=="GraphConvClf":
        vloss_weightvars = [1, 0.01, 10, 10]
        vloss_criterions = [vcriterion_style, vcriterion_sem, vcriterion_func, vcriterion_aesth]
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf2":
        vloss_weightvars = [1, 0.01, 10, 10]
        vloss_criterions = [vcriterion_style, vcriterion_sem, vcriterion_func, vcriterion_aesth]
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf3":
        vloss_weightvars = [1, 0.01, 10, 10]
        vloss_criterions = [vcriterion_style, vcriterion_sem, vcriterion_func, vcriterion_aesth]        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singlesemclass":
        vloss_weightvars = [1, 10, 10]
        vloss_criterions = [vcriterion_style, vcriterion_func, vcriterion_aesth]        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singletask" and not task_flag:
        if _C.SHAPENET_DATA.WHICH_TASK == 'semantic':
            vloss_weightvars = [100]
            vloss_criterions = [vcriterion_sem]
        elif _C.SHAPENET_DATA.WHICH_TASK == 'style':
            vloss_weightvars = [100]
            vloss_criterions = [vcriterion_style]
        elif _C.SHAPENET_DATA.WHICH_TASK == 'functionality':
            vloss_weightvars = [100]
            vloss_criterions = [vcriterion_func]           
        elif _C.SHAPENET_DATA.WHICH_TASK == 'aesthetic':
            vloss_weightvars = [100]
            vloss_criterions = [vcriterion_aesth] 
            
    with torch.no_grad():
        valmodel.eval()
        ##            
        running_vloss_total=0.
        #for i, data in enumerate(tqdm(val_dataloader), 0):
        for i, data in enumerate(val_dataloader, 0):
            label = data[0].cuda()
            mesh = data[1].cuda()
            #batch_prediction = model(mesh)
            outputs = valmodel(mesh)
            #vloss_style = vcriterion_style(outputs[0], label[:,0].long())
            #vloss_semantic = vcriterion_sem(outputs[1], label[:,1].long())
            #vloss_functionality = vcriterion_func(outputs[2], label[:,2].long())
            #vloss_aesthetic = vcriterion_aesth(outputs[3], label[:,3].long())
            ##loss_total = torch.exp(-loss_weightvars[0])*loss_style + \
            ##             torch.exp(-loss_weightvars[1])*loss_semantic + \
            ##             torch.exp(-loss_weightvars[2])*loss_functionality + \
            ##             torch.exp(-loss_weightvars[3])*loss_aesthetic
            #vloss_total = vloss_weightvars[0]*vloss_style + \
            #             vloss_weightvars[1]*vloss_semantic + \
            #             vloss_weightvars[2]*vloss_functionality + \
            #             vloss_weightvars[3]*vloss_aesthetic
            vloss_calc_pertask = torch.stack([losswcrit[0]*losswcrit[1](outputs[ii], label[:,ii].long()) for ii, losswcrit in enumerate(zip(vloss_weightvars, vloss_criterions))])
            vloss_total = torch.sum(vloss_calc_pertask)
            running_vloss_total+=vloss_total.item()
    return running_vloss_total/len(val_dataloader)

def test_model_acc_all(modelpath, val_dataloader, _C):
    # -------------------------------------------------------------
    #   VALIDATION
    # -------------------------------------------------------------
    task_flag = _C.SHAPENET_DATA.WHICH_TASK=='all'

    if _C.GCC.WHICH_GCN_FN=="GraphConvClf":
        valmodel = GraphConvClf(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf2":
        valmodel = GraphConvClf2(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf3":      
        valmodel = GraphConvClf3(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singlesemclass":        
        valmodel = GraphConvClf_singlesemclass(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singletask" and not task_flag:
        if _C.SHAPENET_DATA.WHICH_TASK == 'semantic':
            numCLASSES=2
        elif _C.SHAPENET_DATA.WHICH_TASK == 'style':
            numCLASSES=3
        elif _C.SHAPENET_DATA.WHICH_TASK == 'functionality':        
            numCLASSES=4
        elif _C.SHAPENET_DATA.WHICH_TASK == 'aesthetic':            
            numCLASSES=5
        valmodel = GraphConvClf_singletask(_C, numCLASSES).cuda()
    
    with torch.no_grad():
        #valmodel = GraphConvClf(_C).cuda()
        valmodel.load_state_dict(torch.load(modelpath, map_location=torch.device('cuda:0'))['state_dict'])
        valmodel.eval()
        ##
        #val_acc = 0.0
        #print("\n\n\tEvaluating..")
        correct_style = 0
        correct_semantic = 0
        correct_func = 0
        correct_aesth = 0
        total_style = 0
        total_semantic = 0
        total_func = 0
        total_aesth = 0
        #for i, data in enumerate(tqdm(val_dataloader), 0):
        for i, data in enumerate(val_dataloader, 0):
            label = data[0].cuda()
            mesh = data[1].cuda()
            #batch_prediction = model(mesh)
            outputs = valmodel(mesh)
            # _, predicted = torch.max(outputs.data, 1)
            pred_style =  torch.argmax(outputs[0].data,1)
            pred_semantic = torch.argmax(outputs[1].data,1)
            pred_func = torch.argmax(outputs[2].data,1)
            pred_aesth = torch.argmax(outputs[3].data,1)
            label_style = label[:,0].long()
            label_semantic = label[:,1].long()
            label_func = label[:,2].long()
            label_aesth = label[:,3].long()
            ##
            total_style += label_style.size(0)
            total_semantic += label_semantic.size(0)
            total_func += label_func.size(0)
            total_aesth += label_aesth.size(0)
            ##
            correct_style += int(pred_style == label_style)#.sum().item()
            correct_semantic += int(pred_semantic == label_semantic)#.sum().item()
            correct_func += int(pred_func == label_func)#.sum().item()
            correct_aesth += int(pred_aesth == label_aesth)#.sum().item()
    print('Accuracy of the network on validation model set STYLE: %d %%'%(100 * correct_style / total_style))
    print('Accuracy of the network on validation model set SEMANTIC: %d %%'%(100 * correct_semantic / total_semantic))
    print('Accuracy of the network on validation model set FUNCTIONALITY: %d %%'%(100 * correct_func / total_func))
    print('Accuracy of the network on validation model set AESTHETICS: %d %%'%(100 * correct_aesth / total_aesth))
    
def test_model_acc_singletask(modelpath, val_dataloader, _C):
    # -------------------------------------------------------------
    #   VALIDATION
    # -------------------------------------------------------------
    if _C.SHAPENET_DATA.WHICH_TASK == 'semantic':
            numCLASSES=2
    elif _C.SHAPENET_DATA.WHICH_TASK == 'style':
            numCLASSES=3
    elif _C.SHAPENET_DATA.WHICH_TASK == 'functionality':        
            numCLASSES=4
    elif _C.SHAPENET_DATA.WHICH_TASK == 'aesthetic':            
            numCLASSES=5
    valmodel = GraphConvClf_singletask(_C, numCLASSES).cuda()
    
    with torch.no_grad():
        #valmodel = GraphConvClf(_C).cuda()
        valmodel.load_state_dict(torch.load(modelpath, map_location=torch.device('cuda:0'))['state_dict'])
        valmodel.eval()
        ##
        #val_acc = 0.0
        #print("\n\n\tEvaluating..")
        correct_ = 0
        total_ = 0
        #for i, data in enumerate(tqdm(val_dataloader), 0):
        for i, data in enumerate(val_dataloader, 0):
            label = data[0].cuda()
            mesh = data[1].cuda()
            #batch_prediction = model(mesh)
            outputs = valmodel(mesh)
            # _, predicted = torch.max(outputs.data, 1)
            pred_ = torch.argmax(outputs[0].data,1)
            label_ = label[:,0].long()
            ##
            total_ += label_.size(0)
            ##
            correct_ += int(pred_ == label_)#.sum().item()
    print('Accuracy of the network on validation model set %s: %d %%'%(_C.SHAPENET_DATA.WHICH_TASK, 100 * correct_ / total_))
    
def test_model_acc_singlesemclass(modelpath, val_dataloader, _C):
    # -------------------------------------------------------------
    #   VALIDATION
    # -------------------------------------------------------------
    if _C.GCC.WHICH_GCN_FN=="GraphConvClf":
        valmodel = GraphConvClf(_C).cuda()
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf2":
        valmodel = GraphConvClf2(_C).cuda()
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf3":
        valmodel = GraphConvClf3(_C).cuda()
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singlesemclass":
        valmodel = GraphConvClf_singlesemclass(_C).cuda()
        
    with torch.no_grad():
        #valmodel = GraphConvClf(_C).cuda()
        valmodel.load_state_dict(torch.load(modelpath, map_location=torch.device('cuda:0'))['state_dict'])
        valmodel.eval()
        ##
        vcriterion_style = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
        #vcriterion_sem = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
        vcriterion_func = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
        vcriterion_aesth = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()

        loss_weightvars = [1, 10, 10]
        ##
        #val_acc = 0.0
        #print("\n\n\tEvaluating..")
        correct_style = 0
        #correct_semantic = 0
        correct_func = 0
        correct_aesth = 0
        total_style = 0
        #total_semantic = 0
        total_func = 0
        total_aesth = 0
        #for i, data in enumerate(tqdm(val_dataloader), 0):
        for i, data in enumerate(val_dataloader, 0):
            label = data[0].cuda()
            mesh = data[1].cuda()
            #batch_prediction = model(mesh)
            outputs = valmodel(mesh)
            # _, predicted = torch.max(outputs.data, 1)
            pred_style =  torch.argmax(outputs[0].data,1)
            #pred_semantic = torch.argmax(outputs[1].data,1)
            pred_func = torch.argmax(outputs[1].data,1)
            pred_aesth = torch.argmax(outputs[2].data,1)
            label_style = label[:,0].long()
            #label_semantic = label[:,1].long()
            label_func = label[:,1].long()
            label_aesth = label[:,2].long()
            ##
            total_style += label_style.size(0)
            #total_semantic += label_semantic.size(0)
            total_func += label_func.size(0)
            total_aesth += label_aesth.size(0)
            ##
            correct_style += int(pred_style == label_style)#.sum().item()
            #correct_semantic += int(pred_semantic == label_semantic)#.sum().item()
            correct_func += int(pred_func == label_func)#.sum().item()
            correct_aesth += int(pred_aesth == label_aesth)#.sum().item()
    print('Accuracy of the network on validation model set STYLE: %d %%'%(100 * correct_style / total_style))
    #print('Accuracy of the network on validation model set SEMANTIC: %d %%'%(100 * correct_semantic / total_semantic))
    print('Accuracy of the network on validation model set FUNCTIONALITY: %d %%'%(100 * correct_func / total_func))
    print('Accuracy of the network on validation model set AESTHETICS: %d %%'%(100 * correct_aesth / total_aesth))
    
if __name__ == "__main__":
    
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()
    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)
    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_C.CKP.experiment_path, exist_ok=True)
    _C.dump(os.path.join(_C.CKP.experiment_path, "config.yml"))
    
    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    _C.DEVICE = device
    
    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER & CRITERION
    # --------------------------------------------------------------------------------------------
    ## Datasets
    #trn_objs, val_objs = train_val_split(config=_C)
    #collate_fn = ShapenetDataset.collate_fn
    trn_objs_list, val_objs_list = train_val_split_mesh2aesthetics(config=_C)
    
    collate_fn = mesh2aesthetics_Dataset.collate_fn   

    trn_dataset = mesh2aesthetics_Dataset(_C, trn_objs_list)
    trn_dataloader = DataLoader(trn_dataset, 
                            batch_size=_C.OPTIM.BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    val_dataset = mesh2aesthetics_Dataset(_C, val_objs_list)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=_C.OPTIM.VAL_BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    print("Getting dataset...")
    print("Training Samples: "+str(len(trn_dataloader)))
    print("Validation Samples: "+str(len(val_dataloader)))

    model = GraphConvClf(_C).cuda()
    print(os.path.exists(_C.NETWORK_WEIGHTS_PATH))
    #model = load_network(tmodel, _C.NETWORK_WEIGHTS_PATH
    model.load_state_dict(torch.load(_C.NETWORK_WEIGHTS_PATH, map_location=torch.device('cuda:0'))['state_dict'])
    model.eval()
    args  = {}
    args['EXPERIMENT_NAME'] =  _C.EXPERIMENT_NAME
    args['full_experiment_name'] = _C.CKP.full_experiment_name
    args['experiment_path'] = _C.CKP.experiment_path
    # -------------------------------------------------------------
    #   VALIDATION
    # -------------------------------------------------------------
    #val_acc = 0.0
    print("\n\n\tEvaluating..")
    correct_style = 0
    correct_semantic = 0
    correct_func = 0
    correct_aesth = 0
    total_style = 0
    total_semantic = 0
    total_func = 0
    total_aesth = 0
    #for i, data in enumerate(tqdm(val_dataloader), 0):
    for i, data in enumerate(val_dataloader, 0):
        label = data[0].cuda()
        mesh = data[1].cuda()
        with torch.no_grad():
            #batch_prediction = model(mesh)
            outputs = model(mesh)
            # _, predicted = torch.max(outputs.data, 1)
            pred_style =  torch.argmax(outputs[0].data,1)
            pred_semantic = torch.argmax(outputs[1].data,1)
            pred_func = torch.argmax(outputs[2].data,1)
            pred_aesth = torch.argmax(outputs[3].data,1)
            label_style = label[:,0].long()
            label_semantic = label[:,1].long()
            label_func = label[:,2].long()
            label_aesth = label[:,3].long()
            ##
            total_style += label_style.size(0)
            total_semantic += label_semantic.size(0)
            total_func += label_func.size(0)
            total_aesth += label_aesth.size(0)
            ##
            correct_style += int(pred_style == label_style)#.sum().item()
            correct_semantic += int(pred_semantic == label_semantic)#.sum().item()
            correct_func += int(pred_func == label_func)#.sum().item()
            correct_aesth += int(pred_aesth == label_aesth)#.sum().item()
    print('Accuracy of the network on validation model set STYLE: %d %%'%(100 * correct_style / total_style))
    print('Accuracy of the network on validation model set SEMANTIC: %d %%'%(100 * correct_semantic / total_semantic))
    print('Accuracy of the network on validation model set FUNCTIONALITY: %d %%'%(100 * correct_func / total_func))
    print('Accuracy of the network on validation model set AESTHETICS: %d %%'%(100 * correct_aesth / total_aesth))

