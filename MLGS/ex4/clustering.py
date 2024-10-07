from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def deterministic_eigsh(A, **kwargs):
    np.random.seed(0)
    kwargs['v0'] = np.random.rand(min(A.shape))
    return eigsh(A, **kwargs)


def eigsh_help():
    help(eigsh)


def labels_to_list_of_clusters(z: np.array) -> List[List[int]]:
    """Convert predicted label vector to a list of clusters in the graph.
    This function is already implemented, nothing to do here.
    
    Parameters
    ----------
    z : np.array, shape [N]
        Predicted labels.
        
    Returns
    -------
    list_of_clusters : list of lists
        Each list contains ids of nodes that belong to the same cluster.
        Each node may appear in one and only one partition.
    
    Examples
    --------
    >>> z = np.array([0, 0, 1, 1, 0])
    >>> labels_to_list_of_clusters(z)
    [[0, 1, 4], [2, 3]]
    
    """
    return [np.where(z == c)[0] for c in np.unique(z)]


def construct_laplacian(A: sp.csr_matrix, norm_laplacian: bool) -> sp.csr_matrix:
    """Construct Laplacian of a graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    norm_laplacian : bool
        Whether to construct the normalized graph Laplacian or not.
        If True, construct the normalized (symmetrized) Laplacian, L = I - D^{-1/2} A D^{-1/2}.
        If False, construct the unnormalized Laplacian, L = D - A.
        
    Returns
    -------
    L : scipy.sparse.csr_matrix, shape [N, N]
        Laplacian of the graph.
        
    """
    ##########################################################
    # YOUR CODE HERE
    L = None
    N = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    if norm_laplacian:
        degrees_inv_sqrt = np.power(degrees, -0.5)
        degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(degrees_inv_sqrt)
    
        # Compute the normalized Laplacian L
        I = sp.eye(N, format='csr')
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = D - A
    ##########################################################
    return L


def spectral_embedding(A: sp.csr_matrix, num_clusters: int, norm_laplacian: bool) -> np.array:
    """Compute spectral embedding of nodes in the given graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    num_clusters : int
        Number of clusters to detect in the data.
    norm_laplacian : bool, default False
        Whether to use the normalized graph Laplacian or not.
        
    Returns
    -------
    embedding : np.array, shape [N, num_clusters]
        Spectral embedding for the given graph.
        Each row represents the spectral embedding of a given node.
        The rows have to be sorted in ascending order w.r.t. the corresponding eigenvalues.
    
    """
    if (A != A.T).sum() != 0:
        raise ValueError("Spectral embedding doesn't work if the adjacency matrix is not symmetric.")
    if num_clusters < 2:
        raise ValueError("The clustering requires at least two clusters.")
    if num_clusters > A.shape[0]:
        raise ValueError(f"We can have at most {A.shape[0]} clusters (number of nodes).")

    ##########################################################
    # YOUR CODE HERE
    embedding = None
    L = construct_laplacian(A, norm_laplacian)
    eigenvalues, eigenvectors = deterministic_eigsh(L, k=num_clusters, which='SM')
    # Sort the eigenvectors based on the eigenvalues
    idx = np.argsort(eigenvalues)
    embedding = eigenvectors[:, idx]
    ##########################################################

    return embedding


def compute_ratio_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the ratio cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    ratio_cut : float
        Value of the cut for the given partition of the graph.
        
    """
    
    ##########################################################
    # YOUR CODE HERE
    ratio_cut = None
     # Number of nodes
    N = A.shape[0]
    
    # Find the unique clusters
    clusters = np.unique(z)
    
    # Initialize ratio_cut
    ratio_cut = 0.0
    
    for cluster in clusters:
        # Nodes in the current cluster
        nodes_in_cluster = np.where(z == cluster)[0]
        
        # Nodes not in the current cluster
        nodes_not_in_cluster = np.where(z != cluster)[0]
        
        # Cut size: sum of the edges between nodes in the cluster and nodes not in the cluster
        cut_size = A[nodes_in_cluster][:, nodes_not_in_cluster].sum()
        
        # Size of the current cluster
        cluster_size = len(nodes_in_cluster)
        
        # Ratio cut contribution from this cluster
        if cluster_size > 0:
            ratio_cut += cut_size / cluster_size
            

    ##########################################################
    return ratio_cut


def compute_normalized_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the normalized cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    norm_cut : float
        Value of the normalized cut for the given partition of the graph.
        
    """
    
    ##########################################################
    # YOUR CODE HERE
    norm_cut = None
    # Number of nodes
    N = A.shape[0]
    
    # Find the unique clusters
    clusters = np.unique(z)
    
    # Initialize normalized cut
    norm_cut = 0.0
    
    for cluster in clusters:
        # Nodes in the current cluster
        nodes_in_cluster = np.where(z == cluster)[0]
        
        # Nodes not in the current cluster
        nodes_not_in_cluster = np.where(z != cluster)[0]
        
        # Cut size: sum of the edges between nodes in the cluster and nodes not in the cluster
        cut_size = A[nodes_in_cluster][:, nodes_not_in_cluster].sum()
        
        # Volume of the current cluster
        vol_cluster = A[nodes_in_cluster, :].sum()
        
        # Normalized cut contribution from this cluster
        if vol_cluster > 0:
            norm_cut += cut_size / vol_cluster
    ##########################################################
    return norm_cut
