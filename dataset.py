import os
#dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import scipy
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
from torch.utils.data import Dataset, DataLoader,random_split
torch.set_default_tensor_type(torch.FloatTensor)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,matthews_corrcoef
def fn_y_scdata(y, num_classes, use_cuda, use_float64 = False):
    yp = torch.zeros(y.shape[0], num_classes)

    # send the data to GPU(s)
    if use_cuda:
        yp = yp.cuda()
        y = y.cuda()

    # transform the label y (integer between 0 and 9) to a one-hot
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)

    if use_float64:
        yp = yp.double()
    else:
        yp = yp.float()

    return yp


class SingleCellCached(Dataset):
    
    def __init__(self, data_file, label_file,acc_p,barcode,classnum,use_cuda = False, use_float64 = False):
        super(SingleCellCached).__init__()

        self.data = data_file
        self.data = torch.from_numpy(self.data)
        #########
        #self.labels = self.labels.type(torch.FloatTensor)
        self.data  = self.data.type(torch.FloatTensor)
        self.data  = self.data.cuda()
        self.data = torch.where(torch.isnan(self.data), torch.full_like(self.data, 0),self.data)
        self.data = torch.log(self.data + 1.)
        self.data  = self.data.float()
        self.labels = torch.LongTensor(label_file)
        self.classnum = classnum
        self.labels = fn_y_scdata(self.labels,classnum,use_cuda=True)
        self.acc_p = torch.from_numpy(acc_p)
        self.acc_p  = self.acc_p.type(torch.FloatTensor)
        self.acc_p  = self.acc_p.cuda()
        self.acc_p  = self.acc_p.float()
        self.acc_p = torch.where(torch.isnan(self.acc_p), torch.full_like(self.acc_p, 0),self.acc_p)
        self.barcode = barcode
        self.barcode = torch.tensor(self.barcode)
        self.barcode = self.barcode.cuda()
        self.use_cuda = use_cuda
        self.use_float64 = use_float64 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        xs = self.data[index]
        if self.labels is not None:
            ys = self.labels[index]
        else:
            ys = ""
        acc_p = self.acc_p[index]
        barcode = self.barcode[index]
        return xs, ys,acc_p,barcode
    
 
def setup_data_loader(
    dataset,data_file, label_file,acc_p, barcode,classnum,use_cuda, use_float64,
    batch_size, **kwargs
):

    # instantiate the dataset as training/testing sets
    if "num_workers" not in kwargs:
        kwargs = {"num_workers": 0, "pin_memory": False}

    cached_data = dataset(
        data_file = data_file,  label_file = label_file,acc_p = acc_p,  barcode = barcode,classnum= classnum,use_cuda = use_cuda, use_float64 = use_float64
    )
    
    loader = DataLoader(
        cached_data, batch_size = batch_size, shuffle = True, drop_last=True,**kwargs
    )

    return loader  
    
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train_test_valid_loader_setup(datafile,label_file,acc_p,barcode,classnum,cuda,float64,batch_size,testsize, unsuptrain_size,valid_size
):
    data_loaders = {'sup':None, 'unsup':None, 'valid':None ,'test':None}
    
    #test
    sup_data_filepree ,test_data_file= train_test_split(datafile, test_size=testsize, random_state=42)
    sup_label_filepree, test_label_file = train_test_split(label_file, test_size=testsize, random_state=42)
    sup_acc_ppree ,test_acc_p = train_test_split(acc_p, test_size=testsize, random_state=42)
    sup_barcodepree ,test_barcode = train_test_split(barcode, test_size=testsize, random_state=42)

    sup_data_filepre, unsup_data_file = train_test_split(sup_data_filepree, test_size=unsuptrain_size, random_state=42)
    sup_label_filepre, unsup_label_file = train_test_split(sup_label_filepree, test_size=unsuptrain_size, random_state=42)
    sup_acc_ppre, unsup_acc_p = train_test_split(sup_acc_ppree, test_size=unsuptrain_size, random_state=42)
    sup_barcodepre, unsup_barcode = train_test_split(sup_barcodepree, test_size=unsuptrain_size, random_state=42)

    sup_data_file, valid_data_file = train_test_split(sup_data_filepre, test_size=valid_size, random_state=42)
    sup_label_file, valid_label_file = train_test_split(sup_label_filepre, test_size=valid_size, random_state=42)
    sup_acc_p, valid_acc_p = train_test_split(sup_acc_ppre, test_size=valid_size, random_state=42)    
    sup_barcode, valid_barcode = train_test_split(sup_barcodepre, test_size=valid_size, random_state=42)

    data_loaders['sup'] = setup_data_loader(SingleCellCached, sup_data_file,sup_label_file,sup_acc_p,sup_barcode,classnum, cuda, float64, batch_size
    )

    data_loaders['valid'] = setup_data_loader(SingleCellCached, valid_data_file,valid_label_file,valid_acc_p,valid_barcode,classnum,  cuda, float64, batch_size
    )
    data_loaders['unsup'] = setup_data_loader(SingleCellCached, unsup_data_file,unsup_label_file,unsup_acc_p,unsup_barcode,classnum,  cuda, float64, batch_size
    )
    data_loaders['test'] = setup_data_loader(SingleCellCached, test_data_file,test_label_file,test_acc_p, test_barcode,classnum, cuda, float64, batch_size
    )
    return data_loaders['sup'],data_loaders['valid'],data_loaders['unsup'],data_loaders['test']


def get_accuracy(data_loader, classifier_fn):
        predictions, actuals = [], []

        # use the appropriate data loader
        for (xs, ys,acc,barcode) in data_loader:
            # use classification function to compute all predictions for each batch
            predictions.append(classifier_fn(xs))
            actuals.append(ys)

        # compute the number of accurate predictions
        predictions = torch.cat(predictions, dim=0)
        actuals = torch.cat(actuals, dim=0)
        _, y = torch.topk(actuals, 1)
        _, yhat = torch.topk(predictions, 1)
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        accuracy = accuracy_score(y, yhat)
        f1_macro = f1_score(y, yhat, average='macro')
        f1_weighted = f1_score(y, yhat, average='weighted')
        precision = precision_score(y, yhat, average='macro')
        recall = recall_score(y, yhat, average='macro')
        mcc = matthews_corrcoef(y, yhat)
        ARI = adjusted_rand_score(y.ravel(), yhat.ravel())
        NMI = normalized_mutual_info_score(y.ravel(), yhat.ravel())
        return accuracy, f1_macro, f1_weighted, precision, recall, mcc,ARI,NMI    