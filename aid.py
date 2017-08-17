import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.graph import graph_laplacian

from common import baseline_retrieval



EPS = np.finfo('float32').resolution



## AID ##


def automatic_image_disambiguation(features, queries, select_clusters, gamma = 1.0, k = 200, n_clusters = None, max_clusters = 10):
    """ Automatic Image Disambiguation (our method) based on clustering of directions and directed boni.
    
    features - n-by-d matrix containing d-dimensional features of n samples.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' and 'img_id'. 'img_id' gives the ID of the query
              image and 'relevant' points to a list of IDs of images relevant for this query.
    
    select_clusters - Callback function taking a query dictionary with keys 'relevant' and 'img_id' and a list of lists of images
                      for each cluster as arguments and returning a list of indices of selected clusters.
    
    gamma - Controls the effect of the cluster selection. For gamma < 1.0, the direction of samples must match the selected direction
            more exactly for those samples being adjusted, while for very large gamma, even samples in the orthogonal direction will
            be assigned a highly adjusted distance.
    
    k - The number of baseline retrieval results to be used for the initial clustering step.
    
    n_clusters - The number of clusters (image senses) to be shown to the user for selection of the relevant clusters. If set to None,
                 the number of clusters will be determined heuristically.
    
    max_clusters - Maximum number of clusters. Has only an effect if n_clusters is None.
    
    Returns: re-ranked retrieval results as dictionary mapping query IDs to tuples consisting of an ordered list of retrieved image IDs
             and a corresponding list of adjusted distances to the query.
    """
    
    # Baseline retrieval
    retrievals = baseline_retrieval(features, queries, select_clusters)
    
    for qid, (ret, distances) in retrievals.items():
        
        query = queries[qid]
        query_feat = features[query['img_id']]
        
        # Compute directions from query to results
        directions = features[ret] - query_feat[None,:]
        directions /= np.maximum(np.linalg.norm(directions, axis = -1, keepdims = True), EPS)
        
        # Cluster directions of top results
        nc = n_clusters if (n_clusters is not None) and (n_clusters >= 1) else determine_num_clusters_spectral(directions[:k, :], max_clusters = max_clusters)
        if nc > 1:
            km = KMeans(nc, n_init = 100, max_iter = 1000, n_jobs = -1)
            # The KMeans implementation of sklearn <= 0.18.X suffers from numerical precision errors when using float32,
            # so we convert the data to float64 for clustering. See: https://github.com/scikit-learn/scikit-learn/issues/7705
            cluster_ind = km.fit_predict(directions[:k, :].astype(np.float64))

            # Ask user to select relevant clusters
            cluster_preview = [[id for id, l in zip(ret, cluster_ind) if l == i] for i in range(nc)]
            selected_clusters = select_clusters(query, cluster_preview)

            # Re-rank results by taking their direction in relation to the selected clusters into account
            if (len(selected_clusters) > 0) and (len(selected_clusters) < nc):
                distances = adjust_distances(distances, directions, km.cluster_centers_[selected_clusters, :], gamma)
                ind = np.argsort(distances)
                retrievals[qid] = (ret[ind], distances[ind])
    
    return retrievals


def determine_num_clusters_spectral(X, max_clusters = 10, gamma = None):
    """ Determine number of clusters based on Eigengaps of Graph Laplacian. """
    
    if gamma is None:
        gamma = np.sqrt(X.shape[1])
    
    adjacency = rbf_kernel(X, gamma = gamma)
    laplacian = graph_laplacian(adjacency, normed = True, return_diag = False)
    eig = scipy.linalg.eigh(laplacian, eigvals = (0, min(max_clusters, laplacian.shape[0] - 1)), eigvals_only = True)

    eigengap = eig[1:] - eig[:-1]
    return np.argmax(eigengap) + 1


def adjust_distances(distances, directions, selected_directions, gamma = 1.0):
    """ Reduce distances of samples in the selected directions and increase distances of samples in the opposite directions.
    
    distances - Vector of length n with distances of samples in the database to the query.
    
    directions - n-by-d matrix with directions from the query to samples in the database, normalized to unit length.
    
    selected_directions - m-by-d matrix of relevant directions.
    
    gamma - Controls the effect of the cluster selection. For gamma < 1.0, the direction of samples must match the selected direction
            more exactly for those samples being adjusted, while for very large gamma, even samples in the orthogonal direction will
            be assigned a highly adjusted distance.
    
    Returns: adjusted distances of the samples in the database to the query.
    """
    
    # Broadcast single direction to matrix
    if selected_directions.ndim == 1:
        selected_directions = selected_directions[None,:]
    
    # Normalize directions
    directions = directions / np.maximum(np.linalg.norm(directions, axis = -1, keepdims = True), EPS)
    selected_directions = selected_directions / np.maximum(np.linalg.norm(selected_directions, axis = -1, keepdims = True), EPS)
    
    # Compute cosine similarity to most similar direction as dot product (thanks to normalization)
    sim = np.dot(directions, selected_directions.T).max(axis = -1)
    
    # Fuse distance to query and similarity to directions and re-sort results
    max_dist = np.max(distances)
    return distances - np.sign(sim) * (np.abs(sim) ** gamma) * max_dist



## Hard Cluster Selection on the same clusters as AID ##


def hard_cluster_selection(features, queries, select_clusters, k = 200, n_clusters = None, max_clusters = 10):
    """ Hard Cluster Selection as used by CLUE, but on the clusters determined by AID (our method). """
    
    # Baseline retrieval
    retrievals = baseline_retrieval(features, queries, select_clusters)
    
    for qid, (ret, distances) in retrievals.items():
        
        query = queries[qid]
        query_feat = features[query['img_id']]
        
        # Compute directions from query to results
        directions = features[ret] - query_feat[None,:]
        directions /= np.maximum(np.linalg.norm(directions, axis = -1, keepdims = True), EPS)
        
        # Cluster directions of top results
        nc = n_clusters if (n_clusters is not None) and (n_clusters >= 1) else determine_num_clusters_spectral(directions[:k, :], max_clusters = max_clusters)
        if nc > 1:
            km = KMeans(nc, n_init = 100, max_iter = 1000, n_jobs = -1)
            cluster_ind = km.fit_predict(directions[:k, :].astype(np.float64))

            # Ask user to select relevant clusters
            cluster_preview = [[id for id, l in zip(ret, cluster_ind) if l == i] for i in range(nc)]
            selected_clusters = select_clusters(query, cluster_preview)

            # Put images from the selected clusters first
            retrievals[qid] = (
                np.concatenate(([id for i, id in enumerate(ret[:k]) if cluster_ind[i] in selected_clusters], [id for i, id in enumerate(ret[:k]) if cluster_ind[i] not in selected_clusters], ret[k:])),
                np.concatenate(([dist for i, dist in enumerate(distances[:k]) if cluster_ind[i] in selected_clusters], [dist for i, dist in enumerate(distances[:k]) if cluster_ind[i] not in selected_clusters], distances[k:]))
            )
    
    return retrievals
