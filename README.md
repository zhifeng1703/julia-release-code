# Julia on Manifold Computing

![GitHub License](https://img.shields.io/github/license/zhifeng1703/julia-release-code)
![Julia Version](https://img.shields.io/badge/julia-%3E%3D%201.6.0-blue)

Welcome to the Julia on Manifold Computing repository! This is an under-construction project and it focuses on high performance computing on manifold settings. This repository designs and implements data structures and routine implementations specialized to certain manifolds and algorithms to guarantee efficient execution with accurate timing. The effort goes to the implementations avoiding unsupervised memory allocation, garbage-collection (GC), implicit casting and other features of high level programming languages that could cause severe damage to the execution time with lower-level objects. 

For example, the interfaces of factorization level computation (in openBLAS by default) is limited adn provides no option in overwriting preallocated memories and it cause an unavoidable allocation and a sometime-later GC for the memory of the requested factorization in every call. This severely damages the performance of methods that have major computation distributed in repeated factorizations. This repository make its call to the BLAS and LAPACK library via the low level c-interface `@ccall`. Julia is an actively evolving language and the low level interface has no guarantee on staying the same as it is, so it is recommended to run the julia code in this repository with version 1.6 - 1.9.

## Features

- **Supervised Allocations:** [workspace.jl](https://github.com/zhifeng1703/julia-release-code/blob/main/inc/workspace.jl) provides an simple structure `WSP` that collects and maintains the preallocations memories and all other routines with needs of preallocated workspace would make their own `WSP` to avoid unsupervised allocations and GC.
- **Type Safety:** All codes are written in strong-type-fashion to avoid unsupervised type casting. Due to author's limited knowledge and faith on the pass-by-reference logic on passing sub-array, the Julia-built-in `view` for working on general sub-arrays/matrices is not supported yet. All efficient inplace operation on sub-arrays/matrices will be implemented in specialized routines.
- **Loop Vectorization:** Utilize the loop vectorization functionality provided in [LoopVectorization](https://juliasimd.github.io/LoopVectorization.jl) to achieve optimized native code performance.
- **Supported Manifolds:** Data structures that are specialized to computation primitives in the special orthogonal group, the skew symmetric matrices and the Stiefel manifold are designed and implemented.
- **Efficient Algorithms:** This repository has implemented the following efficient algorithms.
  - *The directional derivative of matrix exponential restricted to the skew symmetric matrices and its inverse actions.*
  - *The BCH method of the endpoint geodesic problem on the Stiefel manifold under the canonical metric.*
  - *The lifting formulation of the endpoint geodesic problem on the Stiefel manfiold under the canonical metric.*
  - *The lifting formulation of the endpoint Grassmann horizontal curve problem on the Stiefel manifold.*
- **Basic Routines:** The following basic routines have their own implementations and/or call interface to the library.
  - The real Schur decomposition on double precision: `LAPACK::DGEES`.
  - The singular value decomposition(SVD) on double precision: `LAPACK::DGESBD`.
  - The LU factorization on double precision: `LAPACK::DGETRF`.
  - The matrix-free generalized minimal residual method (GMRES): `gmres_matfree!`.
