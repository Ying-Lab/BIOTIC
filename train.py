
import matplotlib.pyplot as plt
import os
#dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import scipy

#

import time as tm
import scanpy as sc
import anndata
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import inspect
import pickle
import collections
import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from dataset import setup_seed,SingleCellCached,setup_data_loader,train_test_valid_loader_setup,get_accuracy #
from model import tre
from torch.utils.data import Dataset, DataLoader,random_split
from  custom_mlp import MLP, Exp, ExpM
torch.set_default_tensor_type(torch.FloatTensor)
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam, ExponentialLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_inference_for_epoch(sup_data_loader, unsup_data_loader, losses, use_cuda=True):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(sup_data_loader)
    unsup_batches = len(unsup_data_loader) if unsup_data_loader is not None else 0

    # initialize variables to store loss values
    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(sup_data_loader)
    unsup_iter = iter(unsup_data_loader) if unsup_data_loader is not None else None

    # supervised data
    for i in range(sup_batches):
        # extract the corresponding batch
        (xs, ys, acc_p,barcode) = next(sup_iter)
        if use_cuda:
            xs = xs.cuda()
            #if ys!="":
            if len(ys)>0 and isinstance(ys[0], torch.Tensor):
            #if not isinstance(ys, list):
                ys = ys.cuda() 

        # run the inference for each loss with supervised data as arguments
        for loss_id in range(num_losses):
            new_loss = losses[loss_id].step(xs, ys,acc_p,barcode)
            #if loss_id ==0:print("sup losses basic {}").format(losses[loss_id])
            
            epoch_losses_sup[loss_id] += new_loss

    # unsupervised data
    if unsup_data_loader is not None:
        for i in range(unsup_batches):
            # extract the corresponding batch
            (xs, ys, acc_p,barcode) = next(unsup_iter)

            if use_cuda:
                xs = xs.cuda()
                #if ys!="":
                if len(ys)>0 and isinstance(ys[0], torch.Tensor):
                #if not isinstance(ys, list):
                    ys = ys.cuda() 

            # run the inference for each loss with unsupervised data as arguments
            for loss_id in range(num_losses):
                new_loss = losses[loss_id].step(xs, ys, acc_p,barcode)#step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_evaluate_loss(valid_data_loader, losses, use_cuda=True):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    valid_batches = len(valid_data_loader) if valid_data_loader is not None else 0

    # initialize variables to store loss values
    epoch_losses_unsup = [0.0] * num_losses

    # setup the iterators for training data loaders
    valid_iter = iter(valid_data_loader) if valid_data_loader is not None else None
    # unsupervised data
    if valid_data_loader is not None:
        for i in range(valid_batches):
            # extract the corresponding batch
            (xs, ys, acc_p,barcode) = next(valid_iter)

            if use_cuda:
                xs = xs.cuda()
                #if ys!="":
                if len(ys)>0 and isinstance(ys[0], torch.Tensor):
                #if not isinstance(ys, list):
                    ys = ys.cuda() 

            # run the inference for each loss with unsupervised data as arguments
            for loss_id in range(num_losses):
                new_loss = losses[loss_id].evaluate_loss(xs, ys, acc_p,barcode)#step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_unsup

def lossplot(supplosses,unsupplosses,vallosses):
    temp3=torch.zeros(len(vallosses))
    for i in range(0,len(vallosses)):
        v=vallosses[i].split()
        temp3[i]=float(v[1])
    temp2=torch.zeros(len(unsupplosses))
    for i in range(0,len(unsupplosses)):
        v=unsupplosses[i].split()
        temp2[i]=float(v[1])#0 total 1 avg
    temp1=torch.zeros(len(supplosses))
    for i in range(0,len(supplosses)):
        v=supplosses[i].split()
        temp1[i]=float(v[1])
    plt.plot(temp1, label="sup train losses")
    plt.plot(temp2, label="unsup train losses")
    plt.plot(temp3, label="validation losses")
    plt.xlabel("SVI epoches")
    plt.ylabel("ELBO loss")
    plt.legend()        