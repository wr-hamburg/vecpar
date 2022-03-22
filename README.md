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
  <li> [vecmem library](https://github.com/acts-project/vecmem) </li>
  <li> (optional) OpenMP 5.0 support for CPU backend </li>
  <li> (optional) CUDA 11.5.0 runtime for the GPU backend </li>
  <li> (optional) GoogleTest for running the automated tests </li>
</ul>

## Build & install
`mkdir vecpar-build` <br>
`cmake -S vecpar -B vecpar-build` <br>
`cmake --install vecpar-build`<br>

