import os, sys
import numpy as np
import networkx as nx

# Add sigmasep submodule to path
sys.path.append(os.path.abspath('../submodules/sigmasep/python'))

# Import submodule code
from mSCM import *

def generate_dataset(
        d=4,
        k=2,
        p=0.3,
        m=0,
        add_ind_noise_to_A=True,
        add_ind_noise_to_W=True,
        include_latent=True,
        af=np.tanh,
        sc=1,
        noise='uniform',
        sd=3,
        n=10000,
):
    """
    Function to generate a dataset from a mSCM. Leveraging submodule code.
    args:
        d: number of observed variables
        k: number of latent confounders
        p: probability of edge in graph
        m: about the number of hidden units to add per parent node (excluding rounding error, see submodule code)
        add_ind_noise_to_A: add independent noise to A matrix
        add_ind_noise_to_W: add independent noise to W matrix
        include_latent: include latent variables in the dataset
        af: activation function
        sc: activation function scale
        noise: noise distribution
        sd: noise standard deviation
        n: number of samples
    """
    # Sample a graph and ensure it is cyclic
    while True:
        # Extract edges and adjancency matrix from observed variables
        A = sample_adjacency_matrix(d,k,p,add_ind_noise_to_A)
        # Get edges between observed variables
        V = extract_edges(A)
        if not nx.is_directed_acyclic_graph(nx.from_numpy_array(V, create_using=nx.DiGraph)):
            break
    # Setup NN
    c = num_hidden_units_to_add(A,m)
    # Sample weights and bias
    W, b = sample_weights_and_bias(A,c,add_ind_noise_to_W,include_latent)
    # Sample with mSCM (just observation distribution)
    S = sample_from_mSCM(W,b,np.array([]),n,d,af,sc,sd,noise)
    S = normalranktransform(S)
    return S

print(generate_dataset())