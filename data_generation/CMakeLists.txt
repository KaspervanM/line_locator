cmake_minimum_required(VERSION 3.26)
project(data_generation LANGUAGES CXX CUDA)

# Specify the minimum version of CUDA required
find_package(CUDA REQUIRED)

# Add your CUDA source files
set(CUDA_SOURCE_FILES
        cudaFunctions.cu
)

# Add your regular C++ source files
set(CPP_SOURCE_FILES
        main.cu
)

# Specify your target names for the shared library and the main program
set(LIBRARY_NAME cudaFunctions)
set(APP_NAME main)

# Create the shared library from CUDA source files
cuda_add_library(${LIBRARY_NAME} SHARED ${CUDA_SOURCE_FILES})

# Set the appropriate architecture flags (change this to match your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_90)

# Link the shared library to CUDA libraries
target_link_libraries(${LIBRARY_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Add the include directory for your CUDA header
include_directories(${CUDA_INCLUDE_DIRS})

# Create the executable for your main program
add_executable(${APP_NAME} ${CPP_SOURCE_FILES})

# Link your main program with the shared library
target_link_libraries(${APP_NAME} ${LIBRARY_NAME})

# Add the Google_tests subdirectory
add_subdirectory(Google_tests)
