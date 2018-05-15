# Probabilistic PCA in Python

This repository contains a simple implementation of probabilistic PCA as
introduced in [1].

    [1] Michael E. Tipping and Christopher M. Bishop.
        Probabilistic Principal Component Analysis
        Journal of the Royal Statistical Society. Series B (Statistical Methodology)
        Vol. 61, No. 3 (1999), pp. 611-622.

Also consider citing this master thesis for which this version of probabilistic
PCA was implemented:

    @misc{Stutz2017,
        author = {David Stutz},
        title = {Learning Shape Completion from Bounding Boxes with CAD Shape Priors},
        month = {September},
        year = {2017},
        institution = {RWTH Aachen University},
        address = {Aachen, Germany},
        howpublished = {http://davidstutz.de/},
    }

For theoretical background, consider reading [1], or see the discussion in
Section B.1 of the [master thesis](http://davidstutz.de/projects/shape-completion/#).

## Requirements

Python packages:

* NumPy
* SciPy (specifically `scipy.sparse.linalg.svds`)
* HDF5, i.e. h5py

For visualization:

* Matplotlib

## Usage

To compute a probabilistic PCA, use `ppca_train.py`:

    usage: ppca_train.py [-h] [--input INPUT] [--code CODE]
                         [--approximate_k APPROXIMATE_K] [--mean_file MEAN_FILE]
                         [--V_file V_FILE] [--var_file VAR_FILE]
    
    optional arguments:
      -h, --help            show this help message and exit
      --input INPUT         path input HDF5 file
      --code CODE           size of latent space
      --approximate_k APPROXIMATE_K
                            approximate the variance using approximate_k singular
                            values
      --mean_file MEAN_FILE
                            path to HDF5 mean file
      --V_file V_FILE       path to HDF5 matrix file
      --var_file VAR_FILE   path to HDF5 variance file

The main parameter is the input, which has to be a HDF5 file where the first
dimension is the number of samples, the remaining dimensions do not matter
as they are reshaped. Then, `--code` determines the number of principal
components to use.

As probabilistic PCA requires to compute the variance (see [1]) for which
_all_ eigenvalues are required, the computation can become infeasible for high
dimensionality. Therefore, the variance can be approximated using the first `k`
eigenvalues instead which can be set using `--approximate_k` and should be significantly
larger than `--code` but can also be smaller than the total dimensionality.

The output is stored separately in `--mean_file`, `V_file` and `var_file` --
all HDF5 files.

Using `ppca_test.py`, the computed probabilistic PCA can be tested; for example
on a test or validation set:

    usage: ppca_test.py [-h] [--input INPUT] [--mean_file MEAN_FILE]
                        [--V_file V_FILE] [--var_file VAR_FILE] [--output OUTPUT]
    
    optional arguments:
      -h, --help            show this help message and exit
      --input INPUT         path input HDF5 file
      --mean_file MEAN_FILE
                            path to HDF5 mean file
      --V_file V_FILE       path to HDF5 matrix file
      --var_file VAR_FILE   path to HDF5 variance file
      --output OUTPUT       path to output HDF5 file

Here, the input is a HDF5 file containing the test/validation data.

## Example

As example, we provide a simple dataset of rotated and slightly translated
binary rectangles in `32 x 32` resolution. Probabilistic PCA can be applied
as follows:

    python ppca_train.py --input=/BS/dstutz/work/data/2d/outputs_training_prior_moderate.h5 --code=10

In order to test the decomposition:

    python ppca_test.py --input=/BS/dstutz/work/data/2d/outputs_validation_moderate.h5 --output=predictions.h5

The results can be viewed using:

    python view_hdf5.py --predictions=predictions.h5 --target=/BS/dstutz/work/data/2d/outputs_validation_moderate.h5

## License

Copyright (c) 2018 David Stutz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
