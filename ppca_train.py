import numpy as np
import scipy.sparse.linalg
import os
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path input HDF5 file")
    parser.add_argument("--code", type=int, default=10, help="size of latent space")
    parser.add_argument("--approximate_k", type=int, default=300, help="approximate the variance using approximate_k singular values")
    parser.add_argument("--mean_file", type=str, default='mean.h5', help="path to HDF5 mean file")
    parser.add_argument("--V_file", type=str, default='V.h5', help="path to HDF5 matrix file")
    parser.add_argument("--var_file", type=str, default='var.h5', help="path to HDF5 variance file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('[Error] input file not found')
        exit()
    if args.code <= 0:
        print('[Error] code needs to be larger than zero')
        exit()
    if args.approximate_k < args.code:
        print('[Error] approximate_k needs to be larger or equal to the code')
        exit()

    data = utils.read_hdf5(args.input)
    shape = data.shape
    print('[Training] read ' + args.input)
    print('[Training] shape ' + 'x'.join(map(str, data.shape)))

    # first reshape data from images to vectors
    data = data.reshape(shape[0], np.prod(np.array(shape[1:])))
    print('[Training] reshaped data ' + 'x'.join(map(str, data.shape)))

    # compute mean (is a vector of means per variable)
    mean = np.mean(data.T, axis=1)
    print('[Training] computed mean ' + 'x'.join(map(str, mean.shape)))

    # center
    means = np.repeat(mean.reshape((1, mean.shape[0])), data.shape[0], axis = 0)
    data = data - means
    print('[Training] centered data')

    # We need all the eigenvectors and values ...
    U, s, Vt = scipy.sparse.linalg.svds(data, k=args.code)
    print('[Training] computed first ' + str(args.code) + ' singular vectors')

    approximate_k = min(args.approximate_k, data.shape[0])
    _, s_all, _ = scipy.sparse.linalg.svds(data, k=approximate_k)
    print('[Training] computed first ' + str(approximate_k) + ' singular values')

    # singular values to eigenvalues
    e = s**2/(data.shape[0] - 1)
    e_all = s_all**2/(data.shape[0] - 1)

    # compute variance
    var = 1.0/(data.shape[0] - args.code)*(np.sum(e_all) - np.sum(e))
    print('[Training] variance ' + str(var) + ' (' + str(np.sum(e_all))  + ' / ' + str(np.sum(e)) + ')')

    # compute V
    L_m = np.diag(e - np.ones((args.code))*var)**0.5
    V = Vt.T.dot(L_m)

    mean_file = args.mean_file
    V_file = args.V_file
    var_file = args.var_file

    # transformation is given by V.T*(x - mean), so save V and mean
    utils.write_hdf5(mean_file, mean)
    print('[Training] wrote ' + mean_file)
    utils.write_hdf5(V_file, V)
    print('[Training] wrote ' + V_file)
    utils.write_hdf5(var_file, np.array([var]))
    print('[Training] wrote ' + var_file)
