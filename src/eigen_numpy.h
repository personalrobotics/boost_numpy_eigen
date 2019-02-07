#ifndef _EIGEN_NUMPY_H_
#define _EIGEN_NUMPY_H_

#if PY_VERSION_HEX >= 0x03000000
void* SetupEigenConverters();
void* SetUpEigenTensorConverters();
#else
void SetupEigenConverters();
void SetUpEigenTensorConverters();
#endif

#endif
