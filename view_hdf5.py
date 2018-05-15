import os
import h5py
import argparse
import numpy as np
from matplotlib import pyplot as plt
def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')

    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    tensor = h5f[key][()]
    h5f.close()

    return tensor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize predictions.')
    parser.add_argument('--predictions', type=str, help='Prediction HDF5 file.')
    parser.add_argument('--target', type=str, help='Target HDF5 file.')

    args = parser.parse_args()
    if not os.path.exists(args.predictions):
        print('Predictions file does not exist.')
        exit(1)
    if not os.path.exists(args.target):
        print('Target file does not exist.')
        exit(1)

    predictions = read_hdf5(args.predictions)
    predictions = np.squeeze(predictions)
    print('Read %s.' % args.predictions)

    targets = read_hdf5(args.target)
    targets = np.squeeze(targets)
    print('Read %s.' % args.target)

    for n in range(min(10, predictions.shape[0])):
        plt.clf()
        prediction_file = str(n) + '_prediction.png'
        plt.imshow(predictions[n], vmin=0, vmax=1)
        plt.savefig(prediction_file)

        plt.clf()
        target_file = str(n) + '_target.png'
        plt.imshow(targets[n], vmin=0, vmax=1)
        plt.savefig(target_file)