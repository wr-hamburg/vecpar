# vecpar
This is a header-only library for enabling single source code (C++) to target heterogeneous platforms (CPU, GPU). The project is still in early R&D phase.

## Supported backends
<ul>
  <li> (CPU) OpenMP </li>
  <li> (GPU) CUDA </li>
</ul>

## Dependencies
<ul>
  <li> clang 13 with nvptx support to build the sources </li>
  <li> [vecmem](https://github.com/acts-project/vecmem) </li>
  <li> (optional) OpenMP 5.0 support for CPU backend </li>
  <li> (optional) CUDA 11.5.0 runtime for the GPU backend </li>
  <li> (optional) GoogleTest for running the automated tests </li>
</ul>

## Installation

To build the code:

```sh
cmake  -S vecpar -B <build_directory>
```

```sh
cmake --build <build_directory> \
       -DVECPAR_BUILD_OMP_BACKEND=On \ 
       -DVECPAR_BUILD_CUDA_BACKEND=On 
```

To enable the automated tests, set also `-DVECPAR_BUILD_TESTS=On`.

By default, all build options are enabled.

To install the library:

```sh 
cmake --install <build_directory>
```