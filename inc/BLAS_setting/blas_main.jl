# Default 

using MKL, LinearAlgebra

APPLE_BLAS_PATH = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"

LinearAlgebra.__init__();
MY_BLAS_LIB = "Default(OpenBLAS)";
ACTIVATED_BLAS_PATH = BLAS.get_config().loaded_libs[1].libname;

function SWITCH_TO_OPENBLAS()
    LinearAlgebra.__init__();
    MY_BLAS_LIB = "Default(OpenBLAS)";
    ACTIVATED_BLAS_PATH = BLAS.get_config().loaded_libs[1].libname;
end

function SWITCH_TO_MKL()
    MKL.__init__();
    MY_BLAS_LIB = "MKL";
    ACTIVATED_BLAS_PATH = BLAS.get_config().loaded_libs[1].libname;
end