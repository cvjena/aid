import os.path
from glob import glob



def get_dataset_images(img_dir):
    """ Searches for image files in a given directory.
    
    Files are expected to be named according to the MIRFLICKR format, starting with "im1.jpg" and counting continously.
    
    img_dir - Path of the directory containing the images.
    
    Returns: list of filenames of images found in the directory, sorted by ascending number.
    """
    
    images = []
    i = 1
    while os.path.exists(os.path.join(img_dir, 'im{}.jpg'.format(i))):
        images.append(os.path.join(img_dir, 'im{}.jpg'.format(i)))
        i += 1
    return images


def get_dataset_queries(gt_dir, query_dir, dup_file = None):
    """ Loads pre-defined queries.
    
    gt_dir - Directory containing ground-truth files, named like "<class>_r1.txt". Each file contains a list of
             integral image IDs (counting from 1) that belong to the respective class, one ID per line.
    
    query_dir - Directory containing query files, named like "<class>_query.txt". Each file contains a list of
                integral image IDs (counting from 1) to be used as query for this class, one query per line.
    
    dup_file - Path to a file containing a list of IDs of duplicate images, one list per line. Duplicates of
               the first image on each line will be ignored.
    
    Returns: a dictionary mapping query IDs to dictionaries with keys 'img_id' and 'relevant'. 'img_id' gives
             the ID of the query image and 'relevant' points to a list of IDs of images relevant for this query.
    """
    
    duplicates = {}
    if dup_file is not None:
        with open(dup_file) as df:
            for l in df:
                if l.strip() != '':
                    dup_ids = [int(x) - 1 for x in l.strip().split()]
                    for di in dup_ids[1:]:
                        duplicates[di] = dup_ids[0]
    
    queries = {}
    for query_file in glob(os.path.join(query_dir, '*_query.txt')):
        label_file = os.path.join(gt_dir, os.path.basename(query_file)[:-10] + '_r1.txt')
        if os.path.exists(label_file):
            
            with open(label_file) as lf:
                relevant = set(int(l.strip()) - 1 for l in lf if (l.strip() != '') and ((int(l.strip()) - 1) not in duplicates))
            
            with open(query_file) as qf:
                query_imgs = set(duplicates[int(l.strip()) - 1] if (int(l.strip()) - 1) in duplicates else int(l.strip()) - 1 for l in qf if l.strip() != '')
            
            for qid in query_imgs:
                queries[qid] = { 'img_id' : qid, 'relevant' : relevant, 'ignore' : set(duplicates.keys()) }
    
    return queries


def print_metrics(metrics, tabular = True):
    """ Prints evaluation results.
    
    metrics - Dictionary mapping benchmark names to dictionaries mapping metric names to values.
    
    tabular - If True, results will be printed as table, otherwise in CSV format.
    """
    
    METRICS = ['AP', 'P@1', 'P@10', 'P@50', 'P@100', 'NDCG', 'NDCG@100']
    
    if tabular:
    
        print()
        
        # Print header
        max_name_len = max(len(bench) for bench in metrics.keys())
        print(' | '.join([' ' * max_name_len] + ['{:^6s}'.format(metric) for metric in METRICS]))
        print('-' * (max_name_len + sum(max(len(metric), 6) + 3 for metric in METRICS)))

        # Print result rows
        for bench, results in metrics.items():
            print('{:{}s} | {}'.format(bench, max_name_len, ' | '.join('{:>{}.{}f}'.format(results[metric], max(len(metric), 6), 4) for metric in METRICS)))

        print()
    
    else:
        
        print(';'.join(['Benchmark'] + METRICS))
        for bench, results in metrics.items():
            print('{};{}'.format(bench, ';'.join('{:.7g}'.format(results[metric]) for metric in METRICS)))
