
import os
#model
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from custom_mlp import MLP, Exp
torch.set_default_tensor_type(torch.FloatTensor)
import pyro
import pyro.distributions as dist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class tre(nn.Module):
    def __init__(self,
                 output_size = 10,
                 input_size = 1000,
                 z_dim = 50,
                 hidden_layers = [500,],
                 config_enum = None,
                 mask = None,
                 mask_size = 0,
                 use_cuda = False,
                 aux_loss_multiplier = None,
    ):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.mask = mask
        self.use_cuda = use_cuda
        self.mask = torch.from_numpy(self.mask)
        self.mask_size = mask_size 
        self.aux_loss_multiplier = aux_loss_multiplier
        # define and instantiate the neural networks representing
        # the parameters of various distributions in the model
        self.setup_networks()


    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        # define the neural networks used later in the model and the guide.    
        
        self.encoder_y = MLP(#z->y
            [self.z_dim] + hidden_sizes + [self.output_size],
            activation = nn.Softplus,
            output_activation = nn.Softmax,
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.encoder_z = MLP(#xs->z
            [self.input_size] + hidden_sizes + [[z_dim , z_dim]],
            activation = nn.Softplus,
            output_activation = [None, Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.encoder_ptf = MLP(#xs->ptf
            [self.input_size] + hidden_sizes + [[self.mask_size , self.mask_size ]],
            activation = nn.Softplus,
            output_activation = [None, Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.encoder_tfy = MLP(#tf->y
            [self.mask_size] + hidden_sizes + [self.output_size],
            activation = nn.Softplus,
            output_activation = nn.Softmax,
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.decoder_tf = MLP(#01+mask->tf
            [self.mask_size] + hidden_sizes + [[self.mask_size, self.mask_size]],
            activation =  nn.Softplus,
            output_activation = [None, Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.encoder_ls = MLP(
            [self.input_size ] + hidden_sizes + [[1,1]],
            activation = nn.Softplus,
            output_activation = [nn.Softplus, nn.Softplus],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.decoder_thetas = MLP(
            [self.z_dim + self.output_size + self.mask_size  +self.input_size ] + hidden_sizes + [self.input_size],#
            activation = nn.Softplus,
            output_activation = nn.Sigmoid,
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )
        self.cutoff = nn.Threshold(1.0e-9, 1.0e-9)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.to(device)
        
    def model(self, xs,ys = None, acc_p = None,barcode = None):
        """
        The model corresponds to the following generative process:
        z_TF  ~ Normal(0, I) 		
        y ~ categorical(alpha)
        z_GRN  ~ Normal(Ref, 1) 		
        θ= decoder_θ(z_TF, y, z_GRN, rp,ls)	
        X ~ Multinomial(θ)   	
        :param xs: a batch of vectors of gene counts from a cell
        :param ys: (optional) a batch of the class labels
        :param acc_p: a batch of vectors of gene accessibility score from a cell
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module('scc', self)
        xs=xs.to(device)
        batch_size = xs.size(0)
        options = dict(dtype = xs.dtype, device = xs.device)

        with pyro.plate('data'):
            # if the label y is supervised, sample from the constant prior, 
            # otherwise, observe the value
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (
                1.9 * self.output_size
            )
            y = pyro.sample('y', dist.OneHotCategorical(alpha_prior).to_event(1), obs = ys)
            #sample ztf cell*tf
            prior_loc = torch.zeros(batch_size,self.z_dim, **options)
            prior_scale = torch.ones(batch_size,self.z_dim, **options)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1)).to(device)

            #sample zgrn cell*tf*gene
            tf_loc = torch.ones(batch_size, self.mask_size, **options).cpu()*0.1+(self.mask*0.1).ravel()
            tf_loc = tf_loc.to(device).float()
            tf_scale = torch.ones(batch_size, self.mask_size, **options)
            tf = pyro.sample('tf', dist.Normal(tf_loc, tf_scale).to_event(1)).to(device)
            #sample library_size cell*1
            ls_loc = torch.ones(batch_size, **options)
            ls_scale = torch.ones(batch_size, **options)
            ls = pyro.sample('ls', dist.Weibull(ls_loc, ls_scale)).to(device)
            ls = ls.unsqueeze(-1)####
            
            # true expression
            thetas = self.decoder_thetas([zs,y, tf,acc_p]).to(device)
            thetas = thetas * ls
            thetas = self.cutoff(thetas)

            

            # finally, score the observation
            max_count = torch.ceil(abs(xs).sum(1).sum()).int().item()#有rna的地方读
            # print(max_count)
            pyro.sample('x', dist.DirichletMultinomial(total_count = max_count, concentration = thetas), obs = xs)


    
    def guide(self, xs, ys = None,acc_p = None,barcode = None):
        """
        The guide corresponds to the following:
        μ_(z_TF),Σ_(z_TF)  = encoder_ztf (x)
        z_TF ~ Normal(μ_(z_TF ), Σ_(z_TF ))
        μ_(z_GRN ), Σ_(z_GRN )=encoder_zgrn(x)
        z_GRN  ~ Normal(μ_(z_GRN ), Σ_(z_GRN ))
         y = encoder_y (z_TF)

        :param xs: a batch of vectors of gene counts from a cell
        :param ys: (optional) a batch of the class labels
        :param acc_p: a batch of vectors of gene accessibility score from a cell
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        xs=xs.to(device)
        batch_size = xs.size(0)
        with pyro.plate('data'):
            
            z_loc, z_scale = self.encoder_z(xs)
            zs = pyro.sample('z', dist.Normal(z_loc, z_scale).to_event(1))

            ptf_loc, ptf_scale = self.encoder_ptf(xs)
            tf = pyro.sample('tf', dist.Normal(ptf_loc, ptf_scale).to_event(1))
            

            if ys is None:
                alpha_y = self.encoder_y(zs)
                ys = pyro.sample('y', dist.OneHotCategorical(alpha_y))
            
            y = self.encoder_y(zs)

            ls_loc, ls_scale = self.encoder_ls(xs)
            ls = pyro.sample('ls', dist.Weibull(ls_loc.squeeze(), ls_scale.squeeze()))#20

            
    def classifier(self, xs):
        """
        classify a cell (or a batch of cells)

        :param xs: a batch of vectors of gene counts from a cell
        :return: a batch of the corresponding class labels (as one-hots)
                 along with the class probabilities
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the cell(s)
        z, _ = self.encoder_z(xs)
        alpha = self.encoder_y(z)

        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)

        return ys
    
    def classifier_with_probability(self, xs):
        """
        classify a cell (or a batch of cells)

        :param xs: a batch of vectors of gene counts from a cell
        :return: a batch of the corresponding class labels (as one-hots)
                 along with the class probabilities
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the cell(s)
        z,_ = self.encoder_z(xs)
        alpha = self.encoder_y(z)
        res, ind = torch.topk(alpha, 1)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys, alpha
        
    def predicted_zgrn(self, xs):
        ptf,_ = self.encoder_ptf(xs)
        return ptf
    
    def latent_embedding(self, xs):
        """
        compute the z_scored latent embedding of a cell (or a batch of cells)

        :param xs: a batch of vectors of gene counts from a cell
        :return: a batch of the latent embeddings
        """
        zs,_ = self.encoder_z(xs)
        mu     = torch.mean(zs, axis=0)  
        sigma  = torch.std(zs, axis=0)             
        zs_norm = (zs - mu) / sigma 
        return zs_norm    
    
    def rawlatent_embedding(self, xs):
        """
        compute the latent embedding of a cell (or a batch of cells)

        :param xs: a batch of vectors of gene counts from a cell
        :return: a batch of the latent embeddings
        """
        zs,_ = self.encoder_z(xs)
        return zs
    
    def model_classify(self, xs, ys  = None,acc_p = None,barcode = None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module('scc', self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                zs,_ = self.encoder_z(xs)
                alpha_y = self.encoder_y(zs)
                with pyro.poutine.scale(scale = 50 * self.aux_loss_multiplier):
                    ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(alpha_y), obs = ys)
            

    def guide_classify(self, xs, ys = None,acc_p = None,barcode = None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass
    
    def model_classify1(self, xs, ys  = None,acc_p = None,barcode = None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module('scc', self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                tf,_ = self.encoder_ptf(xs)
                alpha_y = self.encoder_tfy(tf)
                with pyro.poutine.scale(scale = 50 * self.aux_loss_multiplier):
                    ys_aux = pyro.sample('y_aux1', dist.OneHotCategorical(alpha_y), obs = ys)
            

    def guide_classify1(self, xs, ys = None,acc_p = None,barcode = None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass
   
