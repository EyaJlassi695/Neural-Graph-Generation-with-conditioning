# Neural Graph Generation with Conditioning

## Overview
This project is part of the **M2 - ALTEGRAD course** (2024-25 Data Challenge) at École Polytechnique's DaSciM group. The goal is to study and apply machine learning techniques to generate graphs with specific properties, given a text query that describes the desired graph structure.

Graph generation is a challenging machine learning task that has applications in various fields, such as chemo-informatics, where generative models are used to create molecular graphs with specific properties (e.g., high drug-likeness). This challenge focuses on leveraging **latent diffusion models** to conditionally generate graphs.

## Challenge Details
The project consists of training a **Variational Graph Autoencoder (VGAE)** and a **Latent Diffusion Model (LDM)** to:
1. Encode graphs into low-dimensional latent representations.
2. Generate realistic graphs conditioned on text descriptions.

The competition is hosted on [Kaggle](https://www.kaggle.com/competitions/generating-graphs-with-specified-properties/), where participants are tasked with generating graphs and evaluating their accuracy based on specified graph properties.

## Dataset
The dataset contains graphs and their textual descriptions:
- **Train Set**: 8,000 graph-description pairs.
- **Validation Set**: 1,000 graph-description pairs.
- **Test Set**: 1,000 textual descriptions (graphs to be generated).

Graph files are provided in edgelist and GraphML formats, while textual descriptions are in `.txt` format. Data can be downloaded [here](https://drive.google.com/file/d/1Ey54FhVnIUlryhV_AwUFykp4mdjUvcul/view?usp=sharing).

## Baseline Model
The baseline model consists of:
1. **Variational Graph Autoencoder (VGAE)**:
   - Encodes a graph into a latent representation (`z = E(G)`).
   - Decodes the latent representation back into a graph (`G = D(z)`).
2. **Latent Diffusion Model**:
   - Corrupts latent graph representations by adding Gaussian noise.
   - Uses a denoising neural network (`ϵθ`) to predict and remove noise, gradually generating a denoised vector that can be decoded into a graph.

## Evaluation
Generated graphs are evaluated based on **mean absolute error (MAE)** between the generated and actual graph properties. The final submission should be a CSV file containing IDs and predicted graph properties for each test sample.

## Provided Code
The project includes the following scripts:
- `main.py`: Sets up the model, hyperparameters, and training pipeline.
- `autoencoder.py`: Implements the VGAE model.
- `denoiser_model.py`: Implements the diffusion model.
- `utils.py`: Contains utility functions for data preprocessing and parameter setup.
- `extract_feats.py`: Extracts graph properties from textual descriptions.

## Suggested Improvements
Participants are encouraged to explore:
1. Contrastive learning techniques.
2. Alternative encoder and decoder architectures.
3. Incorporation of large language models (LLMs) for text encoding.
4. Experimentation with other generative models, such as GANs, VAE, Normalizing Flows, and autoregressive models.

## Useful Libraries
- **[PyTorch](https://pytorch.org/):** For deep learning model implementation.
- **[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/):** For graph neural networks.
- **[NetworkX](https://networkx.org/):** For graph creation, analysis, and visualization.



