# vecpar
This is a header-only library for enabling single source code (C++) to target heterogeneous platforms (CPU, GPU). The project is still in early R&D phase.

## Supported backends
<ul>
  <li> (CPU) OpenMP </li>
  <li> (GPU) CUDA </li>
  <li> (GPU) OpenMP target - experimental </li>
</ul>

## Dependencies
The project requires LLVM/Clang to build the sources. Recommendation:
```sh
spack install llvm@14.0.0 +all_targets +cuda cuda_arch=<XY>
```

| Dependency                                               | OpenMP backend | CUDA backend | Tests |
|----------------------------------------------------------|---|--------------|-------|
| [vecmem library](https://github.com/acts-project/vecmem) | x | x| x     |
| OpenMP 5.0 (enabled by default with LLVM14)              | x | |       |
| CUDA 11.5.0                                              | | x |       |
| GoogleTest                                               | | | x     |

vecpar uses [nestoroprysk/FunctionComposition](https://github.com/nestoroprysk/FunctionComposition) for supporting the algorithm chaining functionality.

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
