import numpy as np



def baseline_retrieval(features, queries, select_clusters):
    """ Baseline retrieval without disambiguation.
    
    features - n-by-d matrix containing d-dimensional features of n samples.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' and 'img_id'. 'img_id' gives the ID of the query
              image and 'relevant' points to a list of IDs of images relevant for this query.
    
    select_clusters - Not used, only present for compatibility with similar functions.
    
    Returns: Baseline retrieval results as dictionary mapping query IDs to tuples consisting of an ordered list of retrieved image
             IDs and a corresponding list of adjusted distances to the query.
    """
    
    # Build ranked list of retrieval results for each query
    retrievals = {}
    for qid, query in queries.items():
        distances = np.sum((features - features[[query['img_id']],:]) ** 2, axis = 1)
        ranking = np.argsort(distances)
        assert(ranking[0] == query['img_id'])
        retrievals[qid] = (ranking[1:], distances[ranking[1:]])
    return retrievals