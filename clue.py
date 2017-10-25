import numpy as np
import scipy.spatial.distance
from sklearn.cluster import SpectralClustering

import heapq

from common import baseline_retrieval
from utils import tqdm



## CLUE ##

def clue(features, queries, select_clusters, k = 200, max_clusters = 10, T = 0.9, min_cluster_size = 2, show_progress = False):
    """ CLUE method for cluster-based relevance feedback in image retrieval.
    
    Reference:
    Chen, Yixin; Wang, James Z.; Krovetz, Robert.
    "CLUE: Cluster-Based Retrieval of Images by Unsupervised Learning."
    IEEE transactions on Image Processing 14.8, 2005, pp. 1187-1201.
    
    features - n-by-d matrix containing d-dimensional features of n samples.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' and 'img_id'. 'img_id' gives the ID of the query
              image and 'relevant' points to a list of IDs of images relevant for this query.
    
    select_clusters - Callback function taking a query dictionary with keys 'relevant' and 'img_id' and a list of lists of images
                      for each cluster as arguments and returning a list of indices of selected clusters.
    
    k - The number of baseline retrieval results to be used for the initial clustering step.
    
    max_clusters - Maximum number of clusters.
    
    T - Threshold for the n-cut value. Nodes with an n-cut value larger than this threshold won't be subdivided any further.
    
    min_cluster_size - Minimum number of items per cluster.
    
    show_progress - If True, a progress bar will be shown (requires tqdm).
    
    Returns: re-ranked retrieval results as dictionary mapping query IDs to tuples consisting of an ordered list of retrieved image IDs
             and a corresponding list of adjusted distances to the query.
    """
    
    # Baseline retrieval
    retrievals = baseline_retrieval(features, queries, select_clusters)
    
    ret_it = tqdm(retrievals.items(), desc = 'CLUE', total = len(retrievals), leave = False) if show_progress else retrievals.items()
    for qid, (ret, distances) in ret_it:
        
        query = queries[qid]
        query_feat = features[query['img_id']]
        
        # Spectral clustering of top results
        tree = RecursiveNormalizedCuts(max_clusters, T, min_cluster_size)
        tree.fit([(id, features[id]) for id in ret[:k]])
        clusters = tree.clusters()
        
        # Ask user to select relevant clusters
        selected_clusters = select_clusters(query, tree.sort_items_by_centroid_distance())
        
        # Put images from the selected clusters first
        offset = 0
        selected_clusters.sort() # disable cheating through fine-grained relevance ranking
        for c in selected_clusters:
            ret[offset:offset+len(clusters[c])] = [id for id, _ in clusters[c]]
            offset += len(clusters[c])
        
        # Add remaining clusters in tree order
        for i, c in enumerate(clusters):
            if i not in selected_clusters:
                ret[offset:offset+len(c)] = [id for id, _ in c]
                offset += len(c)
    
    return retrievals


class RecursiveNormalizedCuts(object):
    
    def __init__(self, max_clusters, T, min_cluster_size = 2):
        object.__init__(self)
        self.max_clusters = max_clusters
        self.T = T
        self.min_cluster_size = min_cluster_size
        self.tree = { 'depth' : 0, 'height' : 0, 'size' : 0, 'leafs' : 1, 'children' : [], 'parent' : None, 'items' : [], 'affinity' : [] }
    
    
    def fit(self, feat):
        
        # Compute affinity matrix using RBF kernel on pair-wise distances
        affinity = scipy.spatial.distance.pdist(np.array([f for id, f in feat]))
        sigma = -2 * np.var(affinity)
        affinity = np.exp(scipy.spatial.distance.squareform(affinity) / sigma)
        
        # Recursive clustering
        self.tree = { 'depth' : 0, 'height' : 0, 'size' : 0, 'leafs' : 1, 'children' : [], 'parent' : None, 'items' : feat, 'affinity' : affinity }
        queue = []
        heapq.heappush(queue, (-1 * len(self.tree['items']), np.random.rand(), self.tree))
        while (self.tree['leafs'] < self.max_clusters) and (len(queue) > 0):
            if len(queue[0][2]['items']) <= self.min_cluster_size:
                break
            left, right, ncut_value = self.split(heapq.heappop(queue)[2])
            if ncut_value > self.T:
                break
            if (left is not None) and (right is not None):
                heapq.heappush(queue, (-1 * len(left['items']), np.random.rand(), left))
                heapq.heappush(queue, (-1 * len(right['items']), np.random.rand(), right))
    
    
    def split(self, node):
        
        # Perform normalized cut
        try:
            ind = SpectralClustering(2, affinity = 'precomputed', assign_labels = 'discretize').fit_predict(node['affinity'])
        except KeyboardInterrupt:
            raise
        except:
            return None, None, 0
        
        # Create left and right node
        mask1, mask2 = (ind == 0), (ind == 1)
        if not (np.any(mask1) and np.any(mask2)):
            return None, None, 0
        left = { 'depth' : node['depth'] + 1, 'height' : 0, 'size' : 0, 'leafs' : 1, 'children' : [], 'parent' : node, 'items' : [f for i, f in enumerate(node['items']) if ind[i] == 0], 'affinity' : node['affinity'][np.ix_(mask1, mask1)] }
        right = { 'depth' : node['depth'] + 1, 'height' : 0, 'size' : 0, 'leafs' : 1, 'children' : [], 'parent' : node, 'items' : [f for i, f in enumerate(node['items']) if ind[i] == 1], 'affinity' : node['affinity'][np.ix_(mask2, mask2)] }
        
        # Force the node with the lower minimum distance to the query to be the left node
        if ind[0] == 1: # items are already sorted when passed to fit(), so we just need to look at the first item instead of re-computing all distances
            left, right = right, left
        
        # Modify parent
        node['children'] = [left, right]
        
        # Modify parent chain
        parent = node
        while parent is not None:
            parent['height'] += 1
            parent['size'] += 2
            parent['leafs'] += 1
            parent = parent['parent']
        
        return left, right, self.ncut_value(node['affinity'], ind)
    
    
    def clusters(self):
        
        def _clusters(node):
            return sum([_clusters(child) for child in node['children']], []) if len(node['children']) > 0 else [node['items']]
        
        return _clusters(self.tree)
    
    
    def sort_items_by_centroid_distance(self):
        
        clusters = self.clusters()
        sorted_clusters = []
        for c in clusters:
            feat = np.array([f for id, f in c])
            dist = np.linalg.norm(feat - feat.mean(axis = 0), axis = -1)
            ind = np.argsort(dist)
            sorted_clusters.append([c[i][0] for i in ind])
        return sorted_clusters
    
    
    def ncut_value(self, affinity, lbl):
        
        mask_a, mask_b = (lbl == 0), (lbl == 1)
        cut_a_b = affinity[mask_a,:][:,mask_b].sum()
        cut_a_v = affinity[mask_a,:].sum()
        cut_b_v = affinity[mask_b,:].sum()
        if (cut_a_v == 0) or (cut_b_v == 0):
            print(affinity)
            print(lbl)
        return cut_a_b / cut_a_v + cut_a_b / cut_b_v