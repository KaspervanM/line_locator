cmake_minimum_required(VERSION 3.26)
project(data_generation CUDA)

set(CMAKE_CUDA_STANDARD 23)

find_package(CUDAToolkit REQUIRED)

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
add_library(${LIBRARY_NAME} SHARED ${CUDA_SOURCE_FILES})

# Link the shared library to CUDA libraries
target_link_libraries(${LIBRARY_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Create the executable for your main program
add_executable(${APP_NAME} ${CPP_SOURCE_FILES})

# Link your main program with the shared library
target_link_libraries(${APP_NAME} ${LIBRARY_NAME})

# Add the Google_tests subdirectory
add_subdirectory(Google_tests)
