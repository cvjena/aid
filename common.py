import numpy as np
from utils import tqdm



def baseline_retrieval(features, queries, select_clusters, show_progress = False):
    """ Baseline retrieval without disambiguation.
    
    features - n-by-d matrix containing d-dimensional features of n samples.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' and 'img_id'. 'img_id' gives the ID of the query
              image and 'relevant' points to a list of IDs of images relevant for this query.
    
    select_clusters - Not used, only present for compatibility with similar functions.
    
    show_progress - If True, a progress bar will be shown (requires tqdm).
    
    Returns: Baseline retrieval results as dictionary mapping query IDs to tuples consisting of an ordered list of retrieved image
             IDs and a corresponding list of adjusted distances to the query.
    """
    
    # Build ranked list of retrieval results for each query
    retrievals = {}
    query_it = tqdm(queries.items(), desc = 'Baseline', total = len(queries), leave = False) if show_progress else queries.items()
    for qid, query in query_it:
        distances = np.sum((features - features[[query['img_id']],:]) ** 2, axis = 1)
        ranking = np.argsort(distances)
        assert(ranking[0] == query['img_id'])
        retrievals[qid] = (ranking[1:], distances[ranking[1:]])
    return retrievals