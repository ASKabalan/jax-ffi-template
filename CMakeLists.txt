cmake_minimum_required(VERSION 3.19...3.25)


project(NCCL_API_Bench LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit REQUIRED VERSION 12)
message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

# Add the executable
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
set(XLA_DIR ${Python_SITELIB}/jaxlib/include)
message(STATUS "XLA_DIR: ${XLA_DIR}")

# Detect the installed nanobind package and import it into CMake
find_package(nanobind CONFIG REQUIRED)


# Include directories for the executable
set(SOURCES csrc/src/extensions.cc csrc/src/gpu_ops.cu)
nanobind_add_module(gpu_ops STABLE_ABI NB_STATIC ${SOURCES})


target_link_libraries(gpu_ops  PUBLIC CUDA::cudart)
target_include_directories(gpu_ops PUBLIC
                           csrc/include
                           ${XLA_DIR}
                          )

set_target_properties(gpu_ops PROPERTIES CUDA_ARCHITECTURES "70;80;89")
set_target_properties(gpu_ops PROPERTIES
                        LINKER_LANGUAGE CUDA
                        CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS gpu_ops LIBRARY DESTINATION jax_ffi_template_lib)
