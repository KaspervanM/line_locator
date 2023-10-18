cmake_minimum_required(VERSION 3.26)
project(Google_tests)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create the test executable
add_executable(Google_tests_run test.cu)

# Link against gtest_main and CUDAToolkit
target_link_libraries(Google_tests_run gtest_main ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Add a test
add_test(NAME Google_tests_run COMMAND Google_tests_run)
