# vecpar
This is a header-only library for enabling single source code (C++) to target heterogeneous platforms (CPU, GPU). The project is still in early R&D phase.

## Supported backends
<ul>
  <li> (CPU) OpenMP </li>
  <li> (GPU) CUDA </li>
  <li> (GPU) OpenMP target - experimental </li>
</ul>

## Dependencies

Common dependencies for all the backends:
1. [Vecmem library](https://github.com/acts-project/vecmem)
2. [GoogleTest](https://github.com/google/googletest)

LLVM/clang is currently the only compiler that can build all the vecpar backends. Recommended setup on a system with NVIDIA GPU:
```sh
spack install llvm@14.0.0 +all_targets +cuda cuda_arch=<XY>
spack install vecmem +cuda cuda_arch=<XY>
spack install googletest
```  
vecpar uses [nestoroprysk/FunctionComposition](https://github.com/nestoroprysk/FunctionComposition) for supporting the algorithm chaining functionality.

### Dependency for the OpenMP for CPU backend
Any C/C++ compiler (with OpenMP support) can build the CPU OpenMP backend. 

### Dependency for the CUDA backend
For the CUDA backend, `clang` or `nvcc` can be used to compile the code. Additionally, the CUDA libraries must be accessible at compile and runtime.

### Dependency for the OpenMP Target backend
To compile the GPU OpenMP backend, `gcc`/`clang` need a specific build configuration when targeting different GPU:
* NVIDIA - this can be easily achieved by installing `gcc` or `llvm` with flags `+nvptx` and `+cuda` respectively from spack. 
* AMD - the configuration steps need to be done manually as show in the online documentation. For AMD GPU, [AOMP compiler](https://github.com/ROCm-Developer-Tools/aomp) can be used as an alternative.
    
Also, the GPU driver and the associated libraries (CUDA or ROCm) need to be accessible at compile and runtime.

## Installation

Get the code

```sh
git clone --recurse-submodules https://github.com/wr-hamburg/vecpar.git
```

To build the code

```sh
cmake -S <source_dir> -B <build_directory>
```

```sh
cmake --build <build_directory> \
       -DVECPAR_BUILD_OMP_BACKEND=On \ 
       -DVECPAR_BUILD_CUDA_BACKEND=On 
```

To enable the automated tests, set also `-DVECPAR_BUILD_TESTS=On`.

By default, all build options are enabled.

To compile for aarch64, set CC/CXX environment variables to appropriate aarch64 compilers.

To install the library

```sh 
cmake --install <build_directory>
```
## Collection types
vecpar supports `vecmem::vectors` as input and/or output for all operations. 
`vecmem::jagged_vectors` are supported for a restricted subset (marked with x):

| Abstraction | `Jagged_vector` as input(s) | `Jagged_vector` as output |
|-------------|----------------------------|---------------------|
| map | x                          | x                   |
| filter |                            |                     |
| reduce |                            |                     |
| map-filter | x                          |                     |
| map-reduce | x                          |                     |
