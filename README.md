# boost_numpy_eigen [![Build Status](https://travis-ci.org/personalrobotics/boost_numpy_eigen.svg?branch=master)](https://travis-ci.org/personalrobotics/boost_numpy_eigen)

This is a simple example on how to use boost.python to call c++ code from python and convert numpy arrays to [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).

To run (assuming you've boost.python already installed):

```console
$ cmake .
$ make hello
$ python hello.py
$ make test_eigen_numpy_mod
$ python test_eigen_numpy.py
```

## Installation

### On Ubuntu using `apt`

#### Xenial/Bionic

```
$ sudo apt-apt-repository ppa:personalrobotics/ppa
$ sudo apt update
$ sudo apt install python-boost-numpy-eigen
```

## Links

There also is the [`ndarray`](https://github.com/ndarray/ndarray) project, that aims at providing a multidimensionnal
array library similar to numpy.ndarray for C++. Some of the code comes from `ndarray`.

## License

`boost_numpy_eigen` is licensed under a BSD license. See [LICENSE](./LICENSE) for more information.

## Authors

`boost_numpy_eigen` was created by [Julien Rebetez](https://github.com/julienr) ([old repo](https://github.com/julienr/boost_numpy_eigen)). It has received contributions from [Michael Koval](https://github.com/mkoval), [Jeongseok Lee](https://github.com/jslee02), [Pras Velagapudi](https://github.com/psigen), and [Gregory Kramida](https://github.com/Algomorph).
