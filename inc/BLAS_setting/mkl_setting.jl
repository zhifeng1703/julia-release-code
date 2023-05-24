using MKL

MY_BLAS_LIB = "MKL"
ACTIVATED_LIB_PATH = BLAS.get_config().loaded_libs[1].libname
