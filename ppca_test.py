import os
import utils
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path input HDF5 file")
    parser.add_argument("--mean_file", type=str, default='mean.h5', help="path to HDF5 mean file")
    parser.add_argument("--V_file", type=str, default='V.h5', help="path to HDF5 matrix file")
    parser.add_argument("--var_file", type=str, default='var.h5', help="path to HDF5 variance file")
    parser.add_argument("--output", type=str, help="path to output HDF5 file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('[Error] input file not found')
        exit()
    if not os.path.exists(args.mean_file):
        print('[Error] mean file not found')
        exit()
    if not os.path.exists(args.V_file):
        print('[Error] matrix file not found')
        exit()
    if not os.path.exists(args.var_file):
        print('[Error] variance file not found')
        exit()

    data = utils.read_hdf5(args.input)
    shape = data.shape
    print('[Validation] read ' + args.input)

    data = data.reshape((shape[0], np.prod(np.array(shape[1:]))))
    print('[Validation] reshaped data ' + 'x'.join(map(str, data.shape)))

    mean_file = args.mean_file
    V_file = args.V_file
    var_file = args.var_file

    mean = utils.read_hdf5(mean_file)
    print('[Validation] read ' + mean_file)
    V = utils.read_hdf5(V_file)
    print('[Validation] read ' + V_file)
    var = utils.read_hdf5(var_file)[0]
    print('[Validation] read ' + var_file)
    print('[Validation] var is ' + str(var))

    I = np.eye(V.shape[1])
    M = V.T.dot(V) + I*var
    M_inv = np.linalg.inv(M)

    means = np.repeat(mean.reshape((mean.shape[0], 1)), data.shape[0], axis = 1)
    codes = M_inv.dot(V.T.dot(data.T - means))

    code_mean = np.mean(codes)
    code_var = np.var(codes)
    print('[Validation] codes: ' + str(code_mean) + ' / ' + str(code_var))

    preds = np.dot(V, codes) + means
    preds = preds.T

    utils.write_hdf5(args.output, preds.reshape(shape))
    print('[Validation] wrote ' + args.output)