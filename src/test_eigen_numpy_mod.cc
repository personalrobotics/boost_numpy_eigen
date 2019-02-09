// Tests from ndarray lib
#include <boost/python.hpp>
#include <Eigen/Eigen>
#if EIGEN_MAJOR_VERSION > 3 || (EIGEN_MAJOR_VERSION == 3 && EIGEN_MINOR_VERSION > 2)
#include <unsupported/Eigen/CXX11/Tensor>
#endif
#include "eigen_numpy.h"

#include <cstdarg>
#include <iostream>
#include <array>

namespace bp = boost::python;

bool tensorsSupported(){
#if EIGEN_MAJOR_VERSION < 3 || (EIGEN_MAJOR_VERSION == 3 && EIGEN_MINOR_VERSION < 2)
  return false;
#else
  return true;
#endif
}

template <typename T3>
bool acceptTensor3(T3 t){
  return (t(0,0,0) == 1)  && (t(0,0,1) == 2)  && (t(0,0,2) == 3)
    && (t(0,1,0) == 4)  && (t(0,1,1) == 5)  && (t(0,1,2) == 6)

    && (t(1,0,0) == 7)  && (t(1,0,1) == 8)  && (t(1,0,2) == 9)
    && (t(1,1,0) == 10) && (t(1,1,1) == 11) && (t(1,1,2) == 12)

    && (t(2,0,0) == 13) && (t(2,0,1) == 14) && (t(2,0,2) == 15)
    && (t(2,1,0) == 16) && (t(2,1,1) == 17) && (t(2,1,2) == 18)

    && (t(3,0,0) == 19) && (t(3,0,1) == 20) && (t(3,0,2) == 21)
    && (t(3,1,0) == 22) && (t(3,1,1) == 23) && (t(3,1,2) == 24);
}

template <typename T4>
bool acceptTensor4(T4 t){
  return (t(0,0,0,0) == 1)  && (t(0,0,0,1) == 2)  && (t(0,0,0,2) == 3)
    && (t(0,0,1,0) == 4)  && (t(0,0,1,1) == 5)  && (t(0,0,1,2) == 6)

    && (t(0,1,0,0) == 7)  && (t(0,1,0,1) == 8)  && (t(0,1,0,2) == 9)
    && (t(0,1,1,0) == 10) && (t(0,1,1,1) == 11) && (t(0,1,1,2) == 12)

    && (t(0,2,0,0) == 13) && (t(0,2,0,1) == 14) && (t(0,2,0,2) == 15)
    && (t(0,2,1,0) == 16) && (t(0,2,1,1) == 17) && (t(0,2,1,2) == 18)

    && (t(0,3,0,0) == 19) && (t(0,3,0,1) == 20) && (t(0,3,0,2) == 21)
    && (t(0,3,1,0) == 22) && (t(0,3,1,1) == 23) && (t(0,3,1,2) == 24)


    && (t(1,0,0,0) == 25) && (t(1,0,0,1) == 26) && (t(1,0,0,2) == 27)
    && (t(1,0,1,0) == 28) && (t(1,0,1,1) == 29) && (t(1,0,1,2) == 30)

    && (t(1,1,0,0) == 31) && (t(1,1,0,1) == 32) && (t(1,1,0,2) == 33)
    && (t(1,1,1,0) == 34) && (t(1,1,1,1) == 35) && (t(1,1,1,2) == 36)

    && (t(1,2,0,0) == 37) && (t(1,2,0,1) == 38) && (t(1,2,0,2) == 39)
    && (t(1,2,1,0) == 40) && (t(1,2,1,1) == 41) && (t(1,2,1,2) == 42)

    && (t(1,3,0,0) == 43) && (t(1,3,0,1) == 44) && (t(1,3,0,2) == 45)
    && (t(1,3,1,0) == 46) && (t(1,3,1,1) == 47) && (t(1,3,1,2) == 48);
}

template <typename M>
bool acceptMatrix(M m) {
    return (m(0,0) == 1) && (m(0,1) == 2) && (m(0,2) == 3) 
        && (m(1,0) == 4) && (m(1,1) == 5) && (m(1,2) == 6);
}

template <typename M>
bool acceptVector(M m) {
    return (m[0] == 1) && (m[1] == 2) && (m[2] == 3) && (m[3] == 4);
}

template <typename T3>
void fillTensor3(T3 & t){
  t(0,0,0) = 1;t(1,0,0) = 1 + 6;t(2,0,0) = 1 + 12;t(3,0,0) = 1 + 18;
  t(0,0,1) = 2;t(1,0,1) = 2 + 6;t(2,0,1) = 2 + 12;t(3,0,1) = 2 + 18;
  t(0,0,2) = 3;t(1,0,2) = 3 + 6;t(2,0,2) = 3 + 12;t(3,0,2) = 3 + 18;

  t(0,1,0) = 4;t(1,1,0) = 4 + 6;t(2,1,0) = 4 + 12;t(3,1,0) = 4 + 18;
  t(0,1,1) = 5;t(1,1,1) = 5 + 6;t(2,1,1) = 5 + 12;t(3,1,1) = 5 + 18;
  t(0,1,2) = 6;t(1,1,2) = 6 + 6;t(2,1,2) = 6 + 12;t(3,1,2) = 6 + 18;
}

template <typename T3>
T3 returnTensor3() {
    static typename boost::remove_const<typename boost::remove_reference<T3>::type>::type t(4,2,3);
    fillTensor3(t);
    return t;
}

template <typename T4>
void fillTensor4(T4 & t){
  t(0,0,0,0) = 1;t(0,1,0,0) = 1 + 6;t(0,2,0,0) = 1 + 12;t(0,3,0,0) = 1 + 18;
  t(0,0,0,1) = 2;t(0,1,0,1) = 2 + 6;t(0,2,0,1) = 2 + 12;t(0,3,0,1) = 2 + 18;
  t(0,0,0,2) = 3;t(0,1,0,2) = 3 + 6;t(0,2,0,2) = 3 + 12;t(0,3,0,2) = 3 + 18;
  t(0,0,1,0) = 4;t(0,1,1,0) = 4 + 6;t(0,2,1,0) = 4 + 12;t(0,3,1,0) = 4 + 18;
  t(0,0,1,1) = 5;t(0,1,1,1) = 5 + 6;t(0,2,1,1) = 5 + 12;t(0,3,1,1) = 5 + 18;
  t(0,0,1,2) = 6;t(0,1,1,2) = 6 + 6;t(0,2,1,2) = 6 + 12;t(0,3,1,2) = 6 + 18;

  t(1,0,0,0) = 1 + 24;t(1,1,0,0) = 1 + 6 + 24;t(1,2,0,0) = 1 + 12 + 24;t(1,3,0,0) = 1 + 18 + 24;
  t(1,0,0,1) = 2 + 24;t(1,1,0,1) = 2 + 6 + 24;t(1,2,0,1) = 2 + 12 + 24;t(1,3,0,1) = 2 + 18 + 24;
  t(1,0,0,2) = 3 + 24;t(1,1,0,2) = 3 + 6 + 24;t(1,2,0,2) = 3 + 12 + 24;t(1,3,0,2) = 3 + 18 + 24;
  t(1,0,1,0) = 4 + 24;t(1,1,1,0) = 4 + 6 + 24;t(1,2,1,0) = 4 + 12 + 24;t(1,3,1,0) = 4 + 18 + 24;
  t(1,0,1,1) = 5 + 24;t(1,1,1,1) = 5 + 6 + 24;t(1,2,1,1) = 5 + 12 + 24;t(1,3,1,1) = 5 + 18 + 24;
  t(1,0,1,2) = 6 + 24;t(1,1,1,2) = 6 + 6 + 24;t(1,2,1,2) = 6 + 12 + 24;t(1,3,1,2) = 6 + 18 + 24;
}

template <typename T4>
T4 returnTensor4() {
    static typename boost::remove_const<typename boost::remove_reference<T4>::type>::type t(2,4,2,3);
    fillTensor4(t);
    return t;
}

template <typename M>
void fillMatrix(M & m) {
    m(0,0) = 1;
    m(0,1) = 2;
    m(0,2) = 3;
    m(1,0) = 4;
    m(1,1) = 5;
    m(1,2) = 6;
}

template <typename M>
M returnMatrix() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(2,3);
    fillMatrix(m);
    return m;
}

template <typename M>
void fillVector(M & m) {
    m[0] = 1;
    m[1] = 2;
    m[2] = 3;
    m[3] = 4;
}

template <typename M>
M returnVector() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(4);
    fillVector(m);
    return m;
}

template <typename M>
bp::object returnObject() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(2,3);
    fillMatrix(m);
    bp::object o(m);
    return o;
}

static const int X = Eigen::Dynamic;

BOOST_PYTHON_MODULE(test_eigen_numpy_mod) {
  SetupEigenConverters();
  bp::def("acceptMatrix_23d_cref", acceptMatrix< Eigen::Matrix<double,2,3> const &>);
  bp::def("acceptMatrix_X3d_cref", acceptMatrix< Eigen::Matrix<double,X,3> const &>);
  bp::def("acceptMatrix_2Xd_cref", acceptMatrix< Eigen::Matrix<double,2,X> const &>);
  bp::def("acceptMatrix_XXd_cref", acceptMatrix< Eigen::Matrix<double,X,X> const &>);
  bp::def("acceptVector_41d_cref", acceptVector< Eigen::Matrix<double,4,1> const &>);
  bp::def("acceptVector_X1d_cref", acceptVector< Eigen::Matrix<double,X,1> const &>);
  bp::def("acceptVector_14d_cref", acceptVector< Eigen::Matrix<double,1,4> const &>);
  bp::def("acceptVector_1Xd_cref", acceptVector< Eigen::Matrix<double,1,X> const &>);
  bp::def("returnVector_41d", returnVector< Eigen::Matrix<double,4,1> >);
  bp::def("returnVector_14d", returnVector< Eigen::Matrix<double,1,4> >);
  bp::def("returnVector_X1d", returnVector< Eigen::Matrix<double,X,1> >);
  bp::def("returnVector_1Xd", returnVector< Eigen::Matrix<double,1,X> >);
  bp::def("returnMatrix_23d", returnMatrix< Eigen::Matrix<double,2,3> >);
  bp::def("returnMatrix_X3d", returnMatrix< Eigen::Matrix<double,X,3> >);
  bp::def("returnMatrix_2Xd", returnMatrix< Eigen::Matrix<double,2,X> >);
  bp::def("returnMatrix_XXd", returnMatrix< Eigen::Matrix<double,X,X> >);
  bp::def("returnMatrix_23d_c", returnMatrix< Eigen::Matrix<double,2,3> const>);
  bp::def("returnMatrix_X3d_c", returnMatrix< Eigen::Matrix<double,X,3> const>);
  bp::def("returnMatrix_2Xd_c", returnMatrix< Eigen::Matrix<double,2,X> const>);
  bp::def("returnMatrix_XXd_c", returnMatrix< Eigen::Matrix<double,X,X> const>);
  bp::def("returnObject_23d", returnObject< Eigen::Matrix<double,2,3> >);
  bp::def("returnObject_X3d", returnObject< Eigen::Matrix<double,X,3> >);
  bp::def("returnObject_2Xd", returnObject< Eigen::Matrix<double,2,X> >);
  bp::def("returnObject_XXd", returnObject< Eigen::Matrix<double,X,X> >);

  bp::def("tensorsSupported", tensorsSupported);

#if EIGEN_MAJOR_VERSION > 3 || (EIGEN_MAJOR_VERSION == 3 && EIGEN_MINOR_VERSION >= 2)
  bp::def("acceptTensor_423f", acceptTensor3< Eigen::Tensor<float,3> >);
  bp::def("acceptTensor_2423d_cref", acceptTensor4< Eigen::Tensor<double,4> const &>);
  bp::def("acceptTensor_423d_cref", acceptTensor3< Eigen::Tensor<double,3> const &>);
  bp::def("returnTensor_2423d", returnTensor4<Eigen::Tensor<double,4>>);
  bp::def("returnTensor_423d", returnTensor3<Eigen::Tensor<double,3>>);
  bp::def("returnTensor_2423d_c", returnTensor4<Eigen::Tensor<double,4> const>);
  bp::def("returnTensor_423d_c", returnTensor3<Eigen::Tensor<double,3> const>);
#endif
}
