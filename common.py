import numpy as np
from multiprocessing import Pool
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
    query_it = tqdm(queries.keys(), desc = 'Baseline', total = len(queries), leave = False) if show_progress else queries.keys()
    with Pool(initializer = _init_pool, initargs = (features, { qid : query['img_id'] for qid, query in queries.items() })) as p:
        return dict(p.imap_unordered(_retrieval_worker, query_it, 100))


def _init_pool(features, query_img_ids):
    global _feat
    global _img_ids
    _feat = features
    _img_ids = query_img_ids

def _retrieval_worker(qid):
    global _feat
    global _img_ids
    distances = np.sum((_feat - _feat[[_img_ids[qid]],:]) ** 2, axis = 1)
    ranking = np.argsort(distances)
    assert(ranking[0] == _img_ids[qid])
    return (qid, (ranking[1:], distances[ranking[1:]]))
