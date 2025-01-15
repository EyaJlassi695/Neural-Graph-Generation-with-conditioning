# Neural-Graph-Generation-with-conditioning
# Generating Graphs with Specified Properties

## Overview
This project focuses on generating graphs that align with specific properties provided in descriptive textual queries. It was developed as part of the ALTEGRAD 2024 competition, which challenges participants to leverage machine learning models to translate textual descriptions into corresponding graph structures. The objective is to minimize the Mean Absolute Error (MAE) between the generated graphs and target graphs.

## Dataset
The dataset contains:
- 8,000 graph-description pairs for training.
- 1,000 pairs each for validation and testing.

Each textual description outlines key graph properties:
- Number of nodes
- Number of edges
- Average node degree
- Number of triangles
- Clustering coefficient
- Maximum k-core
- Number of communities

The evaluation metric is the Mean Absolute Error (MAE):

\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]
Where:
- \(n\): Number of properties evaluated
- \(y_i\): Target property value
- \(\hat{y}_i\): Generated property value

## Model Architecture

### Components
1. **Text Embedding Module**: Converts descriptions into vector representations using:
   - BERT embeddings (768-dimensional vectors).
   - Gemini API’s text-embedding-004 model for advanced embeddings.

2. **Variational Autoencoder (VAE)**:
   - **Encoder**: Uses a Graph Isomorphism Network (GIN) to create latent representations of graphs.
   - **Decoder**: Constructs the graph’s adjacency matrix using Multi-Layer Perceptrons (MLP) and Gumbel Softmax sampling.
   - Loss function:
     - Reconstruction Loss (MSE)
     - KL Divergence with hyperparameter \(\beta\)

3. **Graph Generation Module**:
   - Combines latent representations and text embeddings to generate graphs.
   - Implements quality control using z-score normalized MAE to select the best graph from generated samples.

### Enhancements
- Added dropout layers to prevent overfitting.
- Expanded dimensions of encoder, decoder, and latent space.
- Optimized hyperparameters using Optuna.
- Improved performance from an initial public MAE score of 0.89274 to 0.07508.

## Implementation
- **Feature Processing**:
  - Extracted numerical features (e.g., node count, clustering coefficient) using regex.
  - Combined numerical and semantic embeddings for rich graph representation.
- **Evaluation**:
  - Used NetworkX and Louvain algorithm for property extraction and community detection.
  - Penalized incomplete graphs to ensure quality.

## Challenges
- Experimented with alternative architectures (e.g., GAT, DCN), which yielded limited improvement.
- Clustering approaches (e.g., k-means) were ineffective due to compact clusters and outliers.

## Future Work
- Implement Generative Adversarial Networks (GANs) to enhance diversity.
- Leverage cloud resources for faster computation.
- Explore hybrid architectures combining GCN, GAT, and DCN.

## How to Run
1. Clone the repository and install dependencies.
   ```bash
   git clone <repository_url>
   cd <repository_name>
   pip install -r requirements.txt
   ```
2. Preprocess the dataset:
   ```bash
   python preprocess.py
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Generate and evaluate graphs:
   ```bash
   python evaluate.py
   ```

## References
[RNS20] Davide Rigoni, Nicolò Navarin, Alessandro Sperduti. "Conditional Constrained Graph Variational Autoencoders for Molecule Design". [arXiv:2009.00725](https://arxiv.org/abs/2009.00725)

