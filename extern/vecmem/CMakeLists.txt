# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building VecMem as part of the vecpar" )

# Declare where to get VecMem from.
set( VECPAR_VECMEM_SOURCE
        "URL;https://github.com/acts-project/vecmem/archive/refs/tags/v0.10.0.tar.gz;URL_MD5;712888e704d2e4c915ac1f25f28da467"
        CACHE STRING "Source for VecMem, when built as part of this project" )
mark_as_advanced(VECPAR_VECMEM_SOURCE )
FetchContent_Declare( VecMem ${VECPAR_VECMEM_SOURCE} )

# Options used in the build of VecMem.
set( VECMEM_BUILD_TESTING FALSE CACHE BOOL
        "Turn off the build of the VecMem unit tests" )

# Get it into the current directory.
FetchContent_MakeAvailable( VecMem )