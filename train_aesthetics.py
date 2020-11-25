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
from style_transfer.utils.torch_utils import train_val_split_mesh2aesthetics, save_checkpoint, accuracy
import matplotlib.pyplot as plt
from test_aesthetics import test_model_loss, test_model_acc_all, test_model_acc_singlesemclass,test_model_acc_singletask


import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("Run training for a particular phase.")
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
    print('number of training samples: '+str(len(trn_objs_list)))
    collate_fn = mesh2aesthetics_Dataset.collate_fn   
    #if _C.OVERFIT:
    #    trn_objs, val_objs = trn_objs[:10], val_objs[:10]
    
    #trn_dataset = ShapenetDataset(_C, trn_objs)
    #print(_C.OPTIM.BATCH_SIZE)
    #pdb.set_trace()
    trn_dataset = mesh2aesthetics_Dataset(_C, trn_objs_list)
    trn_dataloader = DataLoader(trn_dataset, 
                            batch_size=_C.OPTIM.BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    #val_dataset = ShapenetDataset(_C, val_objs)
    val_dataset = mesh2aesthetics_Dataset(_C, val_objs_list)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=_C.OPTIM.VAL_BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    #print("BatchSize: "+str(_C.OPTIM.VAL_BATCH_SIZE))
    print("Training Samples: "+str(len(trn_dataloader)))
    print("Validation Samples: "+str(len(val_dataloader)))

    task_flag = _C.SHAPENET_DATA.WHICH_TASK=='all'

    criterion_style = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_sem = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_func = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    criterion_aesth = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
    
    if _C.GCC.WHICH_GCN_FN=="GraphConvClf":
        loss_weightvars = [1, 0.01, 10, 10]
        loss_criterions = [criterion_style, criterion_sem, criterion_func, criterion_aesth]
        model = GraphConvClf(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf2":
        loss_weightvars = [1, 0.01, 10, 10]
        loss_criterions = [criterion_style, criterion_sem, criterion_func, criterion_aesth]
        model = GraphConvClf2(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf3":
        loss_weightvars = [1, 0.01, 10, 10]
        loss_criterions = [criterion_style, criterion_sem, criterion_func, criterion_aesth]        
        model = GraphConvClf3(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singlesemclass":
        loss_weightvars = [1, 10, 10]
        loss_criterions = [criterion_style, criterion_func, criterion_aesth]        
        model = GraphConvClf_singlesemclass(_C).cuda()
        
    elif _C.GCC.WHICH_GCN_FN=="GraphConvClf_singletask" and not task_flag:
        if _C.SHAPENET_DATA.WHICH_TASK == 'semantic':
            numCLASSES=2
            loss_weightvars = [100]
            loss_criterions = [criterion_sem]
        elif _C.SHAPENET_DATA.WHICH_TASK == 'style':
            numCLASSES=3
            loss_weightvars = [100]
            loss_criterions = [criterion_style]
        elif _C.SHAPENET_DATA.WHICH_TASK == 'functionality':
            loss_weightvars = [100]
            loss_criterions = [criterion_func]           
            numCLASSES=4
        elif _C.SHAPENET_DATA.WHICH_TASK == 'aesthetic':
            loss_weightvars = [100]
            loss_criterions = [criterion_aesth]            
            numCLASSES=5
        model = GraphConvClf_singletask(_C, numCLASSES).cuda()
        
    optimizer = optim.Adam(
        model.parameters(),
        lr=_C.OPTIM.LR,
    )
    
    args  = {}
    args['EXPERIMENT_NAME'] =  _C.EXPERIMENT_NAME
    args['full_experiment_name'] = _C.CKP.full_experiment_name
    args['experiment_path'] = _C.CKP.experiment_path
    args['best_loss'] = _C.CKP.best_loss
    #args['best_acc'] = _C.CKP.best_acc
    
    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    total_step = len(trn_dataloader)
    loss_tracker = []
    print('\n ***************** Training *****************')
    #for epoch in tqdm(range(_C.OPTIM.EPOCH)):
    for epoch in range(_C.OPTIM.EPOCH):
        # --------------------------------------------------------------------------------------------
        #   TRAINING 
        # --------------------------------------------------------------------------------------------
        running_loss = 0.0
        running_loss_pertask = torch.zeros((len(loss_criterions),))
        #running_styleloss=0.
        #runningsem_loss=0.
        #running_funcloss=0.
        #running_aesthloss=0.
        
        model.train()
        #for i, data in enumerate(tqdm(trn_dataloader), 0):
        for i, data in enumerate(trn_dataloader, 0):
            label = data[0].cuda()
            mesh = data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(mesh)
            ##loss = criterion(outputs, label)
            #loss_style = loss_weightvars[0]*criterion_style(outputs[0], label[:,0].long())
            #loss_semantic = loss_weightvars[1]*criterion_sem(outputs[1], label[:,1].long())
            #loss_functionality = loss_weightvars[2]*criterion_func(outputs[2], label[:,2].long())
            #loss_aesthetic = loss_weightvars[3]*criterion_aesth(outputs[3], label[:,3].long())
            #loss_total = loss_style + loss_semantic + loss_functionality + loss_aesthetic
            loss_calc_pertask = torch.stack([losswcrit[0]*losswcrit[1](outputs[ii], label[:,ii].long()) for ii, losswcrit in enumerate(zip(loss_weightvars, loss_criterions))])
            loss_total = torch.sum(loss_calc_pertask)
            loss_total.backward()
            optimizer.step()
            #running_loss += loss.item()
            running_loss += loss_total.item()
            running_loss_pertask +=loss_calc_pertask.cpu().detach().numpy() #.item()
            #running_styleloss+=loss_style.item() #*loss_weightvars[0]
            #runningsem_loss+=loss_semantic.item() #*loss_weightvars[1]
            #running_funcloss+=loss_functionality.item() #*loss_weightvars[2]
            #running_aesthloss+=loss_aesthetic.item() #*loss_weightvars[3]
            ##
        ## print loss at the end of the epoch
        running_loss /= len(trn_dataloader)
        running_loss_pertask /= len(trn_dataloader)
        #running_styleloss /= len(trn_dataloader)
        #runningsem_loss /= len(trn_dataloader)
        #running_funcloss /= len(trn_dataloader)
        #running_aesthloss /= len(trn_dataloader)
        print('\n\t EPOCH %s, totalloss: %s, %s'%(epoch, str(np.around(running_loss, decimals=3)),str(np.around(running_loss_pertask, decimals=3)) ))
        #print('\n\t EPOCH %s, totalloss: %s style: %s sem: %s func: %s aesth: %s'%(epoch, str(np.around(running_loss, decimals=3)), str(np.around(running_styleloss, decimals=3)), str(np.around(runningsem_loss, decimals=3)), str(np.around(running_funcloss, decimals=3)), str(np.around(running_aesthloss, decimals=3))))
        loss_tracker.append(running_loss)
        ## 
        if epoch%10==0: # and epoch!=0:
            print('--------------------------------------------------------------------------------')
            ## print out the validation accuracy
            val_loss = test_model_loss(model, val_dataloader,loss_weightvars,_C)
            ## save model
            args = save_checkpoint(model = model,
                         optimizer  = optimizer,
                         curr_epoch = epoch,
                         curr_loss  = val_loss,
                         curr_step  = (total_step * epoch),
                         args       = args,
                         trn_loss   = running_loss,
                         filename   = ('model@epoch%d.pkl' %(epoch)))
            if _C.SHAPENET_DATA.SEMCLASS=='all' and task_flag:
                test_model_acc_all(os.path.join(args['experiment_path'], 'model@epoch%d.pkl' %(epoch)), val_dataloader, _C)
            elif _C.SHAPENET_DATA.SEMCLASS=='all' and not task_flag:
                test_model_acc_singletask(os.path.join(args['experiment_path'],'model@epoch%d.pkl'%(epoch)), val_dataloader, _C)
            else:
                test_model_acc_singlesemclass(os.path.join(args['experiment_path'],'model@epoch%d.pkl'%(epoch)), val_dataloader, _C)
            print('--------------------------------------------------------------------------------')
        torch.cuda.empty_cache()
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    print('Finished Training')
    fig, ax = plt.subplots()
    ax.plot(loss_tracker)
    ax.set_ylabel('Loss Value')
    ax.set_xlabel('Epochs')
    ax.set_title('Training Loss')
    fig.savefig(os.path.join(_C.CKP.experiment_path, _C.EXPERIMENT_NAME+'_trainingloss.png'))
    ## Final save of the model
    args = save_checkpoint(model      = model,
                         optimizer  = optimizer,
                         curr_epoch = epoch,
                         curr_loss  = val_loss,
                         curr_step  = (total_step * epoch),
                         args       = args,
                         trn_loss   = running_loss,
                         filename   = ('model@epoch%d.pkl' %(epoch)))
    # ----------------------------------------------------------------------------------------
    #   VALIDATION
    # ----------------------------------------------------------------------------------------
    #print("Evaluating on model with lowest loss...")
    #val_loss = test_model_acc(os.path.join(args['experiment_path'], 'model_best_loss.pkl'), val_dataloader, _C)
    if _C.SHAPENET_DATA.SEMCLASS=='all':
        test_model_acc_all(os.path.join(args['experiment_path'], 'model@epoch%d.pkl' %(epoch)), val_dataloader, _C)
    else:
        test_model_acc_singlesemclass(os.path.join(args['experiment_path'], 'model@epoch%d.pkl' %(epoch)), val_dataloader, _C)
    #
    #val_loss = 0.0
    ##val_acc = 0.0
    #print("\n\n\tEvaluating..")
    ##for i, data in enumerate(tqdm(val_dataloader), 0):
    #for i, data in enumerate(val_dataloader, 0):
    #    label = data[0].cuda()
    #    mesh = data[1].cuda()
    #    with torch.no_grad():
    #        #batch_prediction = model(mesh)
    #        outputs = model(mesh)
    #        #loss = criterion(batch_prediction, label)
    #        loss_style = criterion_style(outputs[0], label[:,0].long())
    #        loss_semantic = criterion_sem(outputs[1], label[:,1].long())
    #        loss_functionality = criterion_func(outputs[2], label[:,2].long())
    #        loss_aesthetic = criterion_aesth(outputs[3], label[:,3].long())
    #        #loss_total = torch.exp(-loss_weightvars[0])*loss_style + \
    #        #             torch.exp(-loss_weightvars[1])*loss_semantic + \
    #        #             torch.exp(-loss_weightvars[2])*loss_functionality + \
    #        #             torch.exp(-loss_weightvars[3])*loss_aesthetic
    #        loss_total = loss_style + loss_semantic + loss_functionality + loss_aesthetic
    #        ##acc = accuracy(batch_prediction, label)
    #        #val_loss += loss.item()
    #        val_loss += loss_total.item()
    #        #val_acc += np.sum(acc)
    ## Average out the loss
    #val_loss /= len(val_dataloader)
    ##val_acc /= len(val_dataloader)
    #print('\n\tValidation Loss: '+str(val_loss))
    ##print('\tValidation Acc: '+str(val_acc.item()))
    #print('---------------------------------------------------------------------------------------\n')
    ##print('Best Accuracy on validation',args['best_acc'])
    #print('Best Loss on validation',args['best_loss'])
