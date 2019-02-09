#include <boost/python.hpp>
#include "boost_numpy_eigen/eigen_numpy.h"

BOOST_PYTHON_MODULE(boost_numpy_eigen)
{
  SetupEigenConverters();
}
