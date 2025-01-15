import csv
import networkx as nx
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch_geometric.loader import DataLoader
from extract_feats import extract_numbers_simple
from utils import handle_nan
from community import community_louvain
from tqdm import tqdm

import torch


def precompute_missing(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    y = np.nan_to_num(y, nan=-100.0)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    # Find indices where y is -100
    indices_to_change = np.where(y == -100.0)

    # Set corresponding elements in y and y_pred to 0
    y[indices_to_change] = 0.0
    y_pred[indices_to_change] = 0.0
    zeros_per_column = np.count_nonzero(y, axis=0)

    list_from_array = zeros_per_column.tolist()
    dc = {}
    for i in range(len(list_from_array)):
        dc[i] = list_from_array[i]
    return dc, y, y_pred

def sum_elements_per_column(matrix, dc):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    column_sums = [0] * num_cols

    for col in range(num_cols):
        for row in range(num_rows):
            column_sums[col] += matrix[row][col]

    res = []
    for col in range(num_cols):
        x = column_sums[col]/dc[col]
        res.append(x)

    return res

def compute_graph_properties(G):
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    # Average degree
    degrees = [deg for node, deg in G.degree()]
    avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    # Global clustering coefficient
    global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    # Maximum k-core
    max_k_core = handle_nan(float(max(nx.core_number(G).values())))
    # calculate communities
    partition = community_louvain.best_partition(G)
    num_communities = handle_nan(float(len(set(partition.values()))))
    
    return [num_nodes, num_edges, avg_degree, num_triangles, global_clustering_coefficient, max_k_core, num_communities]

def extract_graph_properties(file_path):
    data = open(file_path, "r")
    graph_properties = []
    for line in data:
        line = line.strip()
        tokens = line.split(",")
        graph_id = tokens[0]
        desc = tokens[1:]
        desc = "".join(desc)
        properties = extract_numbers_simple(desc)
        graph_properties.append(properties);
    return graph_properties

def calculate_mean_std(x):

    sm = [0 for i in range(7)]
    samples = [0 for i in range(7)]

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                sm[i] += it
                samples[i] += 1

    mean = [k / y for k,y in zip(sm, samples)]


    sm2 = [0 for i in range(7)]

    std = []

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                k = (it - mean[i])**2
                sm2[i] += k

    std = [(k / y)**0.5 for k,y in zip(sm2, samples)]
    return mean, std

def z_score_norm_elem(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    mae_st = np.absolute(normalized_true - normalized_gen)

    return np.mean(mae_st)

def z_score_norm(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    dc, normalized_true, normalized_gen = precompute_missing(normalized_true, normalized_gen)


    mae_st = np.absolute(normalized_true - normalized_gen)
    mae = sum_elements_per_column(mae_st, dc)

    return mae

def check(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    dc, normalized_true, normalized_gen = precompute_missing(normalized_true, normalized_gen)


    mae_st = np.absolute(normalized_true - normalized_gen)
    row_means = np.mean(mae_st, axis=1)  # Calculate the mean of each row
    sorted_row_means = np.sort(row_means)  # Sort the means
    return sorted_row_means


def mean_std():
    file_path = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/test/test.txt'
    G1 = extract_graph_properties(file_path)
    mean, std = calculate_mean_std(G1)

    return G1, mean, std

def score():    
    G = []
    with open("output.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile) 
        i = 0
        for row in tqdm(reader): 
            i += 1
            edge_list = eval(row['edge_list'])  # Convert string to list of tuples
            g = nx.Graph()
            g.add_edges_from(edge_list)  # Add edges to the graph
            G.append(compute_graph_properties(g))  # Append the created graph to the list

    file_path = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/test/test.txt'
    print("reading file test : ")
    G1 = extract_graph_properties(file_path)


    mean, std = calculate_mean_std(G1)
    mae = z_score_norm(G, G1, mean, std)

    print("Average Mean Absolute Error:", mae)
    print("Average Mean Absolute Error value:", np.sum(mae) / 7)

def mino_maxo():
    G = []
    with open("output.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile) 
        i = 0
        for row in tqdm(reader): 
            i += 1
            edge_list = eval(row['edge_list'])  # Convert string to list of tuples
            g = nx.Graph()
            g.add_edges_from(edge_list)  # Add edges to the graph
            G.append(compute_graph_properties(g))  # Append the created graph to the list

    file_path = 'C:/Users/JLASSI/Downloads/generating-graphs-with-specified-properties (1)/data/data/test/test.txt'
    print("reading file test : ")
    G1 = extract_graph_properties(file_path)


    mean, std = calculate_mean_std(G1)
    vec = check(G, G1, mean, std)

    print("list:", vec)




# score()
# mino_maxo()

