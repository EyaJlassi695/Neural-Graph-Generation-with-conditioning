import argparse
import ast
import csv
import os
import pickle
import random
import shutil
import warnings
from datetime import datetime

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import construct_nx_from_adj, linear_beta_schedule, preprocess_dataset

warnings.filterwarnings("ignore")


np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=512, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=64, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=5, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=6, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
text_emb_dim=768 # Set the text embedding dimension

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
# Modified: Pass the 'device' to preprocess_dataset
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, device)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, device)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, device)


# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
# Modified: Pass text_emb_dim to the constructor of VariationalAutoEncoder
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, text_emb_dim).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train=0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld  = autoencoder.loss_function(data)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld  = autoencoder.loss_function(data)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
            
        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()

# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
# Modified: Pass text_emb_dim to the constructor of DenoiseNN
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition, text_emb_dim=text_emb_dim).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            # Modified: Send data to the device
            data = data.to(device)
            optimizer.zero_grad()
            x_g, _, _ = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            # Modified: Pass data.text_emb to p_losses
            loss = p_losses(denoise_model, x_g, t, data.text_emb, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()
 
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            # Modified: Send data to the device
            data = data.to(device)
            x_g, _, _ = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            # Modified: Pass data.text_emb to p_losses
            loss = p_losses(denoise_model, x_g, t, data.text_emb, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

del train_loader, val_loader
from evaluation import z_score_norm_elem, mean_std, compute_graph_properties, score

y, mean, std = mean_std()
tries = 1200

# Save to a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    counter = 0
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        # Modified: Send data to the device
        data = data.to(device)
        
        # Modified: Use text_emb as condition
        stat = data.text_emb
        bs = len(data)
        vec = [[] for i in range(len(data))]
        dicto = [{} for i in range(len(data))]
        graph_ids = data.filename
        for it in range(tries):
            # Modified: Pass stat (text_emb) to the sample method
            samples = sample(denoise_model, stat, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
            x_sample = samples[-1]
            # Modified: Pass stat (text_emb) to the decode method
            adj = autoencoder.decode_mu(x_sample, stat)
            for i in range(len(data)):
                    Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())

                    vec[i].append((z_score_norm_elem(list(y[i + counter]), compute_graph_properties(Gs_generated), mean, std), it))
                    dicto[i][it] = Gs_generated   
        for i in range(len(data)):
            vec[i].sort()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in dicto[i][vec[i][0][1]].edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text]) 
        counter += len(data) 

score()

