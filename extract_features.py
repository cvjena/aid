import numpy as np
import caffe
import argparse

from sklearn.decomposition import PCA

from utils import get_dataset_images

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it



def extract_cnn_features(net, mean, img, layer, blob = None):
    """ Extracts image features from a CNN using Caffe.
    
    net - caffe.Net instance
    
    mean - A numpy array with the mean values for each channel.
           Channels must be in BGR order and the scale of values is assumed to be [0,255].
    
    img - The image to extract features from, given either as numpy array in the format
          returned by `caffe.io.load_image` or as the filename of the image.
    
    layer - The name of the layer to forward the net to.
    
    blob - The name of the blob to extract features from. If set to `None`, the name of the layer will be used.
           A list of blob names can be given as well for extracting features from multiple blobs.
    
    Returns: If `imgs` was given as list, a list of results will be returned, otherwise a single result.
             A result is either a list of numpy arrays if `blob` was a list, or a single numpy array otherwise.
    """
    
    # Check parameters
    if blob is None:
        blob = layer
    blobs = [blob] if isinstance(blob, str) else blob
    if layer not in net._layer_names:
        raise RuntimeError('Layer not found: {}'.format(layer))
    for b in blobs:
        if b not in net.blobs:
            raise RuntimeError('Blob not found: {}'.format(b))
    
    # Prepare pre-processor
    transformer = caffe.io.Transformer({'data' : net.blobs['data'].data.shape})
    if mean is not None:
        transformer.set_mean('data', mean)          # subtract the mean channel value
    transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)          # rescale float-valued images from [0,1] to [0,255]
    transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR
    
    # Load image
    img = caffe.io.load_image(img) if isinstance(img, str) else img

    # Convert image to correct data format and copy the result to the input blob
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    # Forward the network and extract features
    net._forward(0, list(net._layer_names).index(layer))
    if isinstance(blob, str):
        return net.blobs[blob].data[0].astype(np.float32)
    else:
        return [net.blobs[b].data[0].astype(np.float32) for b in blobs]



if __name__ == '__main__':
    
    # Set up CLI argument parser
    parser = argparse.ArgumentParser(description = 'Extracts fc6 features from VGG16 for all images in the dataset, followed by PCA.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser_general = parser.add_argument_group('General')
    parser_general.add_argument('--pca_dim', type = int, default = 512, help = 'Number of resulting feature dimensions after PCA.')
    parser_general.add_argument('--feature_dump', type = str, default = 'features.npy', help = 'Filename where the feature matrix will be stored.')
    parser_data = parser.add_argument_group('Data')
    parser_data.add_argument('--img_dir', type = str, default = 'mirflickr/mirflickr', help = 'Directory with the images in the dataset.')
    parser_cnn = parser.add_argument_group('CNN')
    parser_cnn.add_argument('--model', type = str, default = 'model/VGG_ILSVRC_16_layers_deploy.prototxt', help = 'Model file.')
    parser_cnn.add_argument('--weights', type = str, default = 'model/VGG_ILSVRC_16_layers.caffemodel', help = 'Weights file.')
    parser_cnn.add_argument('--mean', type = str, default = 'model/image_mean.txt', help = 'File with channel means.')
    parser_cnn.add_argument('--gpu', type = int, default = None, help = 'GPU device to be used.')
    args = parser.parse_args()
    
    # Load model and image list
    images = get_dataset_images(args.img_dir)
    if len(images) == 0:
        print('Could not find any images. Have you set --img_dir correctly?')
        exit()
    if args.gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(args.model, caffe.TEST, weights = args.weights)
    mean = np.loadtxt(args.mean)
    
    # Extract features
    features = np.array([extract_cnn_features(net, mean, img, 'fc6') for img in tqdm(images, desc = 'Extracting CNN features...', dynamic_ncols = True)], dtype = np.float32)
    
    # Norm, PCA, Norm
    features /= np.linalg.norm(features, axis = 1, keepdims = True)
    features = PCA(args.pca_dim).fit_transform(features)
    features /= np.linalg.norm(features, axis = 1, keepdims = True)
    
    # Save features
    np.save(args.feature_dump, features.astype(np.float32, copy = False))
