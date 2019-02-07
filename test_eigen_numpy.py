
# Tests from ndarray lib
import test_eigen_numpy_mod as eigen_mod
import unittest
import numpy


class TestEigenWrappers(unittest.TestCase):

    def setUp(self):
        self.tensor4_d = numpy.array([[[[ 1,  2,  3], [ 4,  5,  6]],
                                       [[ 7,  8,  9], [10, 11, 12]],
                                       [[13, 14, 15], [16, 17, 18]],
                                       [[19, 20, 21], [22, 23, 24]]],
                                      [[[25, 26, 27], [28, 29, 30]],
                                       [[31, 32, 33], [34, 35, 36]],
                                       [[37, 38, 39], [40, 41, 42]],
                                       [[43, 44, 45], [46, 47, 48]]]],dtype=float)
        self.tensor3_d = numpy.array([[[ 1,  2,  3], [ 4,  5,  6]],
                                      [[ 7,  8,  9], [10, 11, 12]],
                                      [[13, 14, 15], [16, 17, 18]],
                                      [[19, 20, 21], [22, 23, 24]]],dtype=float)
        self.tensor3_f = numpy.array([[[ 1,  2,  3], [ 4,  5,  6]],
                                      [[ 7,  8,  9], [10, 11, 12]],
                                      [[13, 14, 15], [16, 17, 18]],
                                      [[19, 20, 21], [22, 23, 24]]],dtype=numpy.float32)
        self.matrix_i = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        self.matrix_d = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        self.vector_i = numpy.array([1, 2, 3, 4], dtype=int)
        self.vector_d = numpy.array([1, 2, 3, 4], dtype=float)
    
    def testAcceptTensor(self):
        if eigen_mod.tensorsSupported():
            self.assertTrue(eigen_mod.acceptTensor_423f(self.tensor3_f))
            self.assertTrue(eigen_mod.acceptTensor_2423d_cref(self.tensor4_d))
            self.assertTrue(eigen_mod.acceptTensor_423d_cref(self.tensor3_d))

    def testAcceptMatrix(self):
        self.assertTrue(eigen_mod.acceptMatrix_23d_cref(self.matrix_d))
        self.assertTrue(eigen_mod.acceptMatrix_X3d_cref(self.matrix_d))
        self.assertTrue(eigen_mod.acceptMatrix_2Xd_cref(self.matrix_d))
        self.assertTrue(eigen_mod.acceptMatrix_XXd_cref(self.matrix_d))

    def testAcceptVector(self):
        self.assertTrue(eigen_mod.acceptVector_41d_cref(self.vector_d))
        self.assertTrue(eigen_mod.acceptVector_X1d_cref(self.vector_d))
        self.assertTrue(eigen_mod.acceptVector_14d_cref(self.vector_d))
        self.assertTrue(eigen_mod.acceptVector_1Xd_cref(self.vector_d))
        
    def testReturnTensor(self):
        if eigen_mod.tensorsSupported():
            self.assertTrue((eigen_mod.returnTensor_2423d() == self.tensor4_d).all())
            self.assertTrue((eigen_mod.returnTensor_2423d_c() == self.tensor4_d).all())
            self.assertTrue((eigen_mod.returnTensor_423d() == self.tensor3_d).all())
            self.assertTrue((eigen_mod.returnTensor_423d_c() == self.tensor3_d).all())
        
    def testReturnMatrix(self):
        self.assertTrue((eigen_mod.returnMatrix_23d() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_X3d() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_2Xd() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_XXd() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_23d_c() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_X3d_c() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_2Xd_c() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnMatrix_XXd_c() == self.matrix_d).all())

    def testReturnObject(self):
        self.assertTrue((eigen_mod.returnObject_23d() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnObject_X3d() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnObject_2Xd() == self.matrix_d).all())
        self.assertTrue((eigen_mod.returnObject_XXd() == self.matrix_d).all())


if __name__ == "__main__":
    unittest.main()
