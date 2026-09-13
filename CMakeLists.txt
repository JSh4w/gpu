cmake_minimum_required(VERSION 3.24)
project(gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# "native" asks nvcc to target whatever GPU is present on the build machine
# (sm_120 on this Blackwell card). Override per-build if compiling for a
# different target, e.g.: cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Debug builds keep device-side symbols/line info for cuda-gdb and Nsight Compute.
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/kernels/*.cu")

foreach(src ${KERNEL_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
