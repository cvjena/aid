import math



def query_metrics(relevant, retrieved, ignore = []):
    """ Computes several performance metrics for the result of a single query.
    
    relevant - Ground-Truth: Set of IDs relevant to the query.
    
    ret - Ranked list of retrieved image identifiers.
    
    ignore - Optionally, a set of IDs to be ignored, i.e., to be counted neither as true nor as false positive.
    
    Returns: dictionary with the following items:
            - 'AP' : average precision
            - 'P@1' : accuracy regarding the best scoring result
            - 'P@10' : precision among the top 10 results
            - 'P@50' : precision among the top 50 results
            - 'P@100' : precision among the top 100 results
            - 'NDCG' : normalized discounted cumulative gain over the entire list of results
            - 'NDCG@100' : normalized discounted cumulative gain over the top 100 results
    """
    
    return {
        'AP' : average_precision(relevant, retrieved, ignore),
        'P@1' : precision_at_k(1, relevant, retrieved, ignore),
        'P@10' : precision_at_k(10, relevant, retrieved, ignore),
        'P@50' : precision_at_k(50, relevant, retrieved, ignore),
        'P@100' : precision_at_k(100, relevant, retrieved, ignore),
        'NDCG' : ndcg(relevant, retrieved, ignore),
        'NDCG@100' : ndcg(relevant, retrieved, ignore, 100),
    }



def avg_query_metrics(queries):
    """ Computes several performance metrics averaged over multiple queries.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' (a set of relevant image IDs)
              and 'retrieved' (a list of retrieved image IDs).
              In addition, the dictionaries may also contain an item named 'ignore', which gives a set of
              image IDs to be ignored, i.e., to be considered neither a true nor a false positive.
    
    Returns: tuple with 2 items:
        1. dictionary with averages of the performance metrics described at `query_metrics`
        2. dictionary mapping metric names to dictionaries mapping query IDs to values
    """
    
    individual, sums = dict(), dict()
    
    for qid, query in queries.items():
        single = query_metrics(query['relevant'], query['retrieved'], query['ignore'] if 'ignore' in query else [])
        for metric, value in single.items():
            if metric not in individual:
                individual[metric] = dict()
            individual[metric][qid] = value
            if metric not in sums:
                sums[metric] = value
            else:
                sums[metric] += value
    
    for metric, value in sums.items():
        sums[metric] = value / float(len(queries))
    
    return sums, individual



def mean_average_precision(queries):
    """ Computes Mean Average Precision for a set of retrieval results for a number of queries.
    
    queries - Dictionary mapping query IDs to dictionaries with keys 'relevant' (a set of relevant image IDs)
              and 'retrieved' (a list of retrieved image IDs).
              In addition, the dictionaries may also contain an item named 'ignore', which gives a set of
              image IDs to be ignored, i.e., to be considered neither a true nor a false positive.
    
    Returns: mAP
    """
    
    return sum(average_precision(
        query['relevant'],
        query['retrieved'],
        query['ignore'] if 'ignore' in query else []
    ) for qid, query in queries.items()) / len(queries)



def average_precision(relevant, retrieved, ignore = []):
    """ Computes Average Precision for given retrieval results for a particular query.
    
    relevant - Ground-Truth: Set of IDs relevant to the query.
    retrieved - Ranked list of retrieved IDs.
    ignore - Optionally, a set of IDs to be ignored, i.e., to be counted neither as true nor as false positive.
    
    Returns: Area under the precision-recall curve.
    """
    
    ignore = set(ignore)
    relevant = set(relevant) - ignore
    
    # Construct list of ranks of true-positive detections
    ranks = []
    rank = 0
    for rid in retrieved:
        if rid not in ignore:
            if rid in relevant:
                ranks.append(rank)
            rank += 1
    
    return ap_from_ranks(ranks, len(relevant))



def ap_from_ranks(ranks, nres):
    """ Computes Average Precision for retrieval results given by the retrieved ranks of the relevant documents.
    
    ranks - Ordered list of ranks of true positives (zero-based).
    nres  = Total number of positives in dataset.
    
    Returns: Area under the precision-recall curve.
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        precision_0 = 1.0 if rank == 0 else ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp+1) / float(rank+1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap



def precision_at_k(k, relevant, retrieved, ignore = []):
    """ Computes the precision among the top `k` retrieval results for a particular query.
    
    k - Number of best-scoring results to be considered. Instead of a single number, a list of `k` values may be given
        to evaluate precision at. In this case, this function will return a list of corresponding precisions as well.
    relevant - Ground-Truth: Set of IDs relevant to the query.
    retrieved - Ranked list of retrieved IDs.
    ignore - Optionally, a set of IDs to be ignored, i.e., to be counted neither as true nor as false positive.
    
    Returns: Precision@k, given as a single number or a list, depending on the value of `k`
    """
    
    ks = [k] if isinstance(k, int) else list(k)
    ks.sort()
    k_index = 0
    
    precisions = []
    
    rank, tp = 0, 0
    for ret in retrieved:
        if ret not in ignore:
            if ret in relevant:
                tp += 1
            rank += 1
            while (k_index < len(ks)) and (rank >= ks[k_index]):
                precisions.append(float(tp) / float(rank))
                k_index += 1
            if k_index >= len(ks):
                break
    return precisions[0] if isinstance(k, int) else precisions



def ndcg(relevant, retrieved, ignore = [], k = None):
    """ Computes the Normalized Discounted Cumulative Gain of a list of retrieval results with binary relevance levels for a particular query.
    
    relevant - Ground-Truth: Set of IDs relevant to the query.
    retrieved - Ranked list of retrieved IDs.
    ignore - Optionally, a set of IDs to be ignored, i.e., to be counted neither as true nor as false positive.
    k - Optionally, the maximum number of best-scoring results to be considered (will compute the NDCG @ k).
        If set to `None`, the NDCG of the entire list of retrieval results will be computed.
        Instead of a single number, a sorted list of values for various `k` may be given to evaluate NDCG at.
        In this case, this function will return a list of corresponding precisions as well.
    
    Returns: NDCG@k, given as a single number or a list, depending on the value of `k`
    """
    
    ignore = set(ignore)
    max_rank = len(set(retrieved) - ignore)
    
    ks = [k] if isinstance(k, int) or (k is None) else list(k)
    for i in range(len(ks)):
        if (ks[i] is None) or (ks[i] > max_rank):
            ks[i] = max_rank
    ks.sort()
    k_index = 0
    
    ndcgs = []
    
    rank, cgain, normalizer = 0, 0.0, 0.0
    for ret in retrieved:
        if ret not in ignore:
            rank += 1
            gain = 1.0 / math.log2(rank + 1)
            if ret in relevant:
                cgain += gain
            if rank <= len(relevant):
                normalizer += gain
            while (k_index < len(ks)) and (rank >= ks[k_index]):
                ndcgs.append(cgain / normalizer)
                k_index += 1
            if k_index >= len(ks):
                break
    return ndcgs[0] if isinstance(k, int) or (k is None) else ndcgs