Automatic Query Image Disambiguation (AID)
==========================================

This repository contains the reference implementation of AID and code that can be used
to reproduce the results from the corresponding [paper][4]:

> Björn Barz and Joachim Denzler.  
> "Automatic Query Image Disambiguation for Content-based Image Retrieval."  
> International Conference on Computer Vision Theory and Applications (VISAPP), 2018.

If you use AID, please cite that paper.


What is AID?
------------

![aid-schema](https://user-images.githubusercontent.com/7915048/31986052-52dd0688-b967-11e7-84d0-c778aa129f3d.png)

AID is a novel recommendation and re-ranking technique for content-based image retrieval (CBIR).
Query images presented to a CBIR system are usually ambiguous, so that user feedback is crucial for
refining the search results towards the actual search objective pursued by the user.

Instead of asking the user to mark multiple relevant and irrelevant images among the initial search
results one by one, AID automatically discovers different meanings of the query image by clustering
the top search results and then asks the users to simply select the cluster that seems most relevant
to them. This way, the user's effort is minimized.

Many similar methods restrict the set of refined results to the selected cluster. However, such an
approach is sub-optimal, because the final set of results will be a strict subset of the initial
set used for clustering, though there could be more relevant images not present among the first
top results. AID; in contrast, applies a global re-ranking of all images in the database with
respect to both the cluster selected by the user and the similarity to the initial query image.

For details, please refer to the paper mentioned above.


Dependencies
------------

##### Mandatory

- `Python >= 3.3`
- `numpy`
- `scipy`
- `scikit-learn`

##### Optional

- `caffe` & `pycaffe` (required if you want to extract the image features yourself)
- `tqdm` (for progress bars)
- `matplotlib` (if you would like to generate graphs for Precision@k)


Reproducing the results from the paper
--------------------------------------

### Getting the features

Before you can actually run the benchmark of the different query image disambiguation methods,
you need to compute some features for the images in the dataset. You can either just download
a [.npy file with pre-computed features][1] (49 MB) for the MIRFLICKR dataset or you can extract
the features yourself as follows:

1. Download the [MIRFLICKR-25K dataset][2] (2.9 GB).
2. Extract the downloaded file inside of the `mirflickr` directory of this repository, so that you
   end up with another `mirflickr` directory inside of the top-level `mirflickr` directory.
3. Download the [pre-trained weights of the VGG 16 model][3] (528 MB) and store them in the `model`
   directory.
4. From the root directory of the repository, run: `python extract_features.py`

### Running the benchmark

Once you have downloaded or extracted the features of the dataset images, you can run the benchmark
as follows:

    python evaluate_query_disambiguation.py --plot_precision

See `python evaluate_query_disambiguation.py --help` for the full list of options.

The result should be similar to the following:

                |   AP   |  P@1   |  P@10  |  P@50  | P@100  |  NDCG  | NDCG@100
    ----------------------------------------------------------------------------
    Baseline    | 0.4201 | 0.6453 | 0.6191 | 0.5932 | 0.5790 | 0.8693 |   0.5869
    CLUE        | 0.4221 | 0.8301 | 0.7829 | 0.6466 | 0.5978 | 0.8722 |   0.6306
    Hard-Select | 0.4231 | 0.8138 | 0.8056 | 0.6773 | 0.6116 | 0.8727 |   0.6450
    AID         | 0.5188 | 0.8263 | 0.7950 | 0.7454 | 0.7212 | 0.8991 |   0.7351

The baseline results should match exactly, while deviations may occur in the other rows due to
randomization.

However, running the benchmark on the entire MIRFLICKR-25K dataset might take about a week and lots of RAM.
If you would like to perform a slightly faster consistency check, you can also run the evaluation on
a set of 70 pre-defined queries (5 for each topic):

    python evaluate_query_disambiguation.py --query_dir mirflickr --rounds 10 --show_sd --plot_precision

In that case, the results should be similar to:

                |   AP   |  P@1   |  P@10  |  P@50  | P@100  |  NDCG  | NDCG@100
    ----------------------------------------------------------------------------
    Baseline    | 0.3753 | 0.7286 | 0.6800 | 0.6100 | 0.5664 | 0.8223 |   0.5880
    CLUE        | 0.3810 | 0.9100 | 0.8133 | 0.6462 | 0.5816 | 0.8290 |   0.6232
    Hard-Select | 0.3849 | 0.8457 | 0.8469 | 0.6846 | 0.6011 | 0.8314 |   0.6426
    AID         | 0.4625 | 0.8757 | 0.8206 | 0.7211 | 0.6711 | 0.8531 |   0.6991
    
    
    Standard Deviation:
    
                |   AP   |  P@1   |  P@10  |  P@50  | P@100  |  NDCG  | NDCG@100
    ----------------------------------------------------------------------------
    Baseline    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |   0.0000
    CLUE        | 0.0005 | 0.0239 | 0.0074 | 0.0045 | 0.0033 | 0.0003 |   0.0037
    Hard-Select | 0.0006 | 0.0270 | 0.0068 | 0.0072 | 0.0031 | 0.0005 |   0.0039
    AID         | 0.0053 | 0.0203 | 0.0087 | 0.0085 | 0.0088 | 0.0017 |   0.0075



[1]: http://www.inf-cv.uni-jena.de/dbvmedia/de/Barz/AID/features.npy
[2]: http://press.liacs.nl/mirflickr/mirflickr25k/mirflickr25k.zip
[3]: http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
[4]: http://hera.inf-cv.uni-jena.de:6680/pdf/Barz18:AID.pdf
