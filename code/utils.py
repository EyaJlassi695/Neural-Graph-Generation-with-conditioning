# utils.py
import csv
import math
import os
import re

import community as community_louvain
import google.generativeai as genai
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import torch
import torch.nn.functional as F

# from extract_feats import extract_feats, extract_numbers # Removed this import
from extract_feats import extract_feats
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from grakel.utils import graph_from_networkx
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Set your Gemini API key
genai.configure(api_key="AIzaSyCgjQ1U99cSdF-91ERBxZtm9cG8VOllzxI")

def text_to_embedding(text, model_name="models/text-embedding-004"):
    """
    Encodes the given text into an embedding vector using the Gemini model.

    Args:
        text (str): The input text to be embedded.
        model_name (str, optional): The Gemini embedding model to use.
            Defaults to "models/text-embedding-004".
    Returns:
        torch.Tensor: A tensor representing the text embedding.
    """
    def extract_numbers(text):
        # Use regular expression to find integers and floats
        numbers = re.findall(r'\d+\.\d+|\d+', text)
        # Convert the extracted numbers to float
        return [float(num) for num in numbers]
    try:
        result = genai.embed_content(model=model_name, content=text, output_dimensionality=761)
        embedding_list = result['embedding']
        graph_features = extract_numbers(text)
        nodes = (graph_features[0] - 30.649600)/1.381798e+02
        edges = (graph_features[1] - 225.407500)/5.473379e+04
        average_degree = (graph_features[2] - 12.887920)/1.039890e+02
        triangles = (graph_features[3] - 1384.119100)/7.763409e+06
        global_clustering_coefficient = (graph_features[4] - 0.504624)/1.042195e-01
        max_k_core = (graph_features[5] - 11.409500)/1.000798e+02
        communities = (graph_features[6] - 3.368900)/2.161029e+00
        
        graph_features = [nodes, edges, average_degree, triangles, global_clustering_coefficient, max_k_core, communities]
        embedding_list += graph_features
        if isinstance(embedding_list, list):
             embedding = torch.tensor(embedding_list, dtype=torch.float32)
        else:
             embedding = torch.tensor(embedding_list, dtype=torch.float32).unsqueeze(0)
        return embedding
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        return torch.zeros(768).unsqueeze(0) # Or any other default return if needed


def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, device):
    data_lst = []
    if dataset == 'test':
        filename = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/dataset_'+dataset+'.pt'
        desc_file = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                
                # Embed the textual description
                feats_text_emb = text_to_embedding(desc).to(device) # Added to device
                data_lst.append(Data(text_emb=feats_text_emb, filename = graph_id))
            fr.close()                    
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/dataset_'+dataset+'.pt'
        graph_path = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/'+dataset+'/graph'
        desc_path = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)
                
                feats_text = extract_feats(fstats)

                # Embed the textual description
                feats_text_emb = text_to_embedding(feats_text).to(device)
                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, text_emb=feats_text_emb, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start