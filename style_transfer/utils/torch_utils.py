import torch
import os
import shutil
import pandas as pd, random
import numpy as np
import logging
from tqdm import tqdm
import pickle
import pdb
import csv



def train_val_split(config, ratio=0.7):
    '''
    Function for splitting dataset in train and validation
    '''
    print("Splitting Dataset..")
    data_dir = config.SHAPENET_DATA.PATH 
    taxonomy = pd.read_json(data_dir+'/taxonomy.json')
    classes = [i for i in os.listdir(data_dir) if i in '0'+taxonomy.synsetId.astype(str).values]
    random.shuffle(classes)
    assert classes != [], "No  objects(synsetId) found."
    if config.OVERFIT:
        classes.remove('04401088')
        classes = classes[:10]
    classes = dict(zip(classes,np.arange(len(classes))))
    
    ## Save the class to synsetID mapping
    path = os.path.join(config.CKP.experiment_path, 'class_map.pkl') 
    with open(path, 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trn_objs = []
    val_objs = []

    for cls in tqdm(classes):
        tmp = [(classes[cls], os.path.join(data_dir, cls, obj_file,'model.obj')) for obj_file in os.listdir(os.path.join(data_dir,cls))]
        random.shuffle(tmp)
        tmp_train = tmp[:int(len(tmp)*0.7)]
        tmp_test = tmp[int(len(tmp)*0.7):]
        trn_objs += tmp_train
        val_objs += tmp_test
        print(taxonomy['name'][taxonomy.synsetId == int(cls)], len(tmp))
    random.shuffle(trn_objs)
    random.shuffle(val_objs)
    return trn_objs, val_objs

def train_val_split_mesh2aesthetics(config, ratio=0.7):
    '''
    Function for splitting dataset in train and validation
    '''
    def labels2vec(labelparams):
        ## FILL in
        return vec_
    print("Splitting Dataset..")
    #data_dir = os.path.join(config.SHAPENET_DATA.PATH, 'Separated')
    data_dir = os.path.join(config.SHAPENET_DATA.PATH)
    _params_csv = os.path.join(config.SHAPENET_DATA.PATH, 'NamingBookV3.csv')

    ## read in params
    tmp_objs = []
    tester_func=[]
    tester_aesth=[]
    STYLECLASSESDICT={'baroque':0, 'modern':1,'moden':1, 'classic':1, '(Insert Label)':1, 'cubist':2, 'cubims':2, 'cu':2, 'cubism':2, 'Cubism':2}
    SEMANTICCLASSESDICT={'house':0, 'House':0, 'column':1, 'Column':1}
    with open(_params_csv, newline='') as csvfile:
        sreader = csv.reader(csvfile, delimiter=',')
        ## skip first two lines of csv file
        next(sreader)
        next(sreader)
        for row in sreader:
            #print(row[0])
            #if 'set36' in row[0] or 'house21' in row[0]:
            if np.any([rr=='' for rr in row]) or np.any([rr=='#' for rr in row]):
                ## skip empty lines or models with incomplete labels
                continue
            objpath = os.path.join(data_dir,row[0])
            unproc_params = [pp for pp in row[1:5]]
            funcp = int(unproc_params[2]) if int(unproc_params[3]) <5 else 5
            aesthp= int(unproc_params[3]) if int(unproc_params[3]) <5 else 5
            params  = [STYLECLASSESDICT[unproc_params[0]], SEMANTICCLASSESDICT[unproc_params[1]], funcp, aesthp ]
            #params = [float(pp.replace(',','')) for pp in row[1:5]]
            tmp_objs.append([params, objpath])
            tester_func.append(int(unproc_params[2]))
            tester_aesth.append(int(unproc_params[3]))
            #print( objpath, params, unproc_params)
    print(np.max(tester_func), np.min(tester_func), np.unique(tester_func))
    print(np.max(tester_aesth), np.min(tester_aesth), np.unique(tester_aesth))
    #print(len(tmp_objs))
    #pdb.set_trace()
    trn_objs = tmp_objs[:int(len(tmp_objs)*0.9)]
    val_objs = tmp_objs[int(len(tmp_objs)*0.9):]
    #for cls in tqdm(classes):
    #    tmp = [(classes[cls], os.path.join(data_dir, cls, obj_file,'model.obj')) for obj_file in os.listdir(os.path.join(data_dir,cls))]
    #    random.shuffle(tmp)
    #    tmp_train = tmp[:int(len(tmp)*0.7)]
    #    tmp_test = tmp[int(len(tmp)*0.7):]
    #    trn_objs += tmp_train
    #    val_objs += tmp_test
    #    #print(taxonomy['name'][taxonomy.synsetId == int(cls)], len(tmp))
    random.shuffle(trn_objs)
    random.shuffle(val_objs)
    return trn_objs, val_objs

def train_val_split_mesh2acoust(config, ratio=0.7):
    '''
    Function for splitting dataset in train and validation
    '''
    print("Splitting Dataset..")
    data_dir = os.path.join(config.SHAPENET_DATA.PATH, 'OBJdatabase')
    acoustic_params_csv = os.path.join(config.SHAPENET_DATA.PATH, 'AcousticParameters.csv')
    #taxonomy = pd.read_json(data_dir+'/taxonomy.json')
    #classes = [i for i in os.listdir(data_dir) if i in '0'+taxonomy.synsetId.astype(str).values]
    #random.shuffle(classes)
    #assert classes != [], "No  objects(synsetId) found."
    #if config.OVERFIT:
    #    classes.remove('04401088')
    #    classes = classes[:10]
    #classes = dict(zip(classes,np.arange(len(classes))))
    #
    ## Save the class to synsetID mapping
    #path = os.path.join(config.CKP.experiment_path, 'class_map.pkl') 
    #with open(path, 'wb') as handle:
    #    pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## read in acoustic params
    tmp_objs = []
    with open(acoustic_params_csv, newline='') as csvfile:
        sreader = csv.reader(csvfile, delimiter=',')
        ## skip first two lines of csv file
        next(sreader)
        next(sreader)
        for row in sreader:                
            objpath = os.path.join(data_dir,row[0])
            params = [float(pp.replace(',','')) for pp in row[1:11]]
            tmp_objs.append([params, objpath])
            #pdb.set_trace()
    trn_objs = tmp_objs[:int(len(tmp_objs)*0.9)]
    val_objs = tmp_objs[int(len(tmp_objs)*0.9):]
    #for cls in tqdm(classes):
    #    tmp = [(classes[cls], os.path.join(data_dir, cls, obj_file,'model.obj')) for obj_file in os.listdir(os.path.join(data_dir,cls))]
    #    random.shuffle(tmp)
    #    tmp_train = tmp[:int(len(tmp)*0.7)]
    #    tmp_test = tmp[int(len(tmp)*0.7):]
    #    trn_objs += tmp_train
    #    val_objs += tmp_test
    #    #print(taxonomy['name'][taxonomy.synsetId == int(cls)], len(tmp))
    random.shuffle(trn_objs)
    random.shuffle(val_objs)
    return trn_objs, val_objs

#def save_checkpoint(model, optimizer, curr_epoch, curr_step, args, curr_loss, curr_acc, trn_loss, filename ):
#    """
#        Saves a checkpoint and updates the best loss and best weighted accuracy
#    """
#    is_best_loss = curr_loss < args['best_loss']
#    is_best_acc = curr_acc > args['best_acc']#
#
#    args['best_acc'] = max(args['best_acc'], curr_acc)
#    args['best_loss'] = min(args['best_loss'], curr_loss)#
#
#    state = {   'epoch':curr_epoch,
#                'step': curr_step,
#                'args': args,
#                'state_dict': model.state_dict(),
#                'val_loss': curr_loss,
#                'val_acc': curr_acc,
#                'trn_loss':trn_loss,
#                'best_val_loss': args['best_loss'],
#                'best_val_acc': args['best_acc'],
#                'optimizer' : optimizer.state_dict(),
#             }
#    path = os.path.join(args['experiment_path'], filename)
#    torch.save(state, path)
#    if is_best_loss:
#        shutil.copyfile(path, os.path.join(args['experiment_path'], 'model_best_loss.pkl'))
#    if is_best_acc:
#        shutil.copyfile(path, os.path.join(args['experiment_path'], 'model_best_acc.pkl'))#
#
#    return args

def save_checkpoint(model, optimizer, curr_epoch, curr_step, args, curr_loss, trn_loss, filename ):
    """
        Saves a checkpoint and updates the best loss and best weighted accuracy
    """
    is_best_loss = curr_loss < args['best_loss']
    args['best_loss'] = min(args['best_loss'], curr_loss)
    #
    state = {   'epoch':curr_epoch,
                'step': curr_step,
                'args': args,
                'state_dict': model.state_dict(),
                'val_loss': curr_loss,
                'trn_loss':trn_loss,
                'best_val_loss': args['best_loss'],
                'optimizer' : optimizer.state_dict(),
             }
    path = os.path.join(args['experiment_path'], filename)
    torch.save(state, path)
    if is_best_loss:
        shutil.copyfile(path, os.path.join(args['experiment_path'], 'model_best_loss.pkl'))
        #
    return args

def accuracy(output, target, topk=(1,)):
    """ From The PyTorch ImageNet example """
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
