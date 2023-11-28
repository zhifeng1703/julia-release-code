using LinearAlgebra
using MKL

mkl_int = Int64
mkl_double = Float64

lapack_int = Int64
lapack_logical = Int64
lapack_double = Float64
lapack_double_complex = ComplexF64

LAPACK_ROW_MAJOR = 101
LAPACK_COL_MAJOR = 102


BLAS_LIBRARIES = ["Built-in openBLAS", "MKL"]





function init_openBLAS()
    LinearAlgebra.__init__()
    global BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname
    if BLAS_LOADED_LIB_IND != 1
        global BLAS_LOADED_LIB_IND = 1
        println("Loaded BLAS library:\t$(BLAS_LIBRARIES[BLAS_LOADED_LIB_IND])")
    end
    return nothing
end

function init_MKL()
    MKL.__init__()
    global BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname
    if BLAS_LOADED_LIB_IND != 2
        global BLAS_LOADED_LIB_IND = 2
        println("Loaded BLAS library:\t$(BLAS_LIBRARIES[BLAS_LOADED_LIB_IND])")
    end
    return nothing
end

BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname;


# println("Initialize BLAS library: \t$(BLAS_LIBRARIES[2])")
MKL.__init__() #By default, use built-in openBLAS

BLAS_LOADED_LIB_IND = 2;


