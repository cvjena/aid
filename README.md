Automatic Query Image Disambiguation (AID)
==========================================

This repository contains the reference implementation of AID and code that can be used
to reproduce the results from the corresponding paper:

> Björn Barz and Joachim Denzler.
> "Automatic Query Image Disambiguation for Content-based Image Retrieval."

If you use AID, please cite that paper.


Dependencies
------------

##### Mandatory

- Python >= 3.3
- numpy
- scipy
- scikit-learn

##### Optional

- caffe & pycaffe (required if you want to extract the image features yourself)
- tqdm (for progress bars during feature extraction)
- matplotlib (if you would like to generate graphs for Precision@k)


Reproducing the results from the paper
--------------------------------------

### Getting the features

Before you can actually run the benchmark of the different query image disambiguation methods,
you need to compute some features for the images in the dataset. You can either just download
a [.npy file with pre-computed features][1] (49 MB) for the MIRFLICKR dataset or you can extract
the features yourself as follows:

1. Download the MIRFLICKR-25K dataset:
   http://press.liacs.nl/mirflickr/mirflickr25k/mirflickr25k.zip (2.9 GB)
2. Extract the downloaded file inside of the `mirflickr` directory of this directory, so that you
   end up with another `mirflickr` directory inside of the top-level `mirflickr` directory.
3. Download the pre-trained weights of the VGG 16 model and store them in the `model` directory:
   http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel (528 MB)
4. From the root directory of the repository, run: `python extract_features.py`

### Running the benchmark

Once you have downloaded or extracted the features of the dataset images, you can run the benchmark
as follows:

    python evaluate_query_disambiguation.py --show_sd --plot_precision

See `python evaluate_query_disambiguation.py --help` for the full list of options.

The result should be similar to the following:

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

The baseline results should match exactly, while deviations may occur in the other rows due to
randomization.



[1]: http://www.inf-cv.uni-jena.de/dbvmedia/de/Barz/AID/features.npy
