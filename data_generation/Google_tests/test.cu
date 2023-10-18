#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include "../cudaFunctions.cu"

TEST(TestMapRange, mapRange) {
    uint8_t input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint8_t *d_arr;
    cudaMalloc(&d_arr, 9 * sizeof(uint8_t));
    cudaMemcpy(d_arr, input, 9 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    const uint8_t min = 2;
    const uint8_t max = 5;

    // Call the function
    mapRange(d_arr, 9, min, max);

    // Copy the result back to the host
    uint8_t result[9];
    cudaMemcpy(result, d_arr, 9 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Check the result
    for (auto num: result) {
        ASSERT_LE(num, max);
        ASSERT_GE(num, min);
    }
}

TEST(TestMapValidLineSegments, mapValidLineSegments) {
    const uint16_t minLength = 5;
    const uint16_t width = 28;
    const uint16_t height = 28;
    const uint32_t nrLines = 5;
    ushort2 input[nrLines * 2] = {{0,                                0},
                                  {0,                                0},

                                  {static_cast<uint16_t>(width - 1), static_cast<uint16_t>(height - 1)},
                                  {static_cast<uint16_t>(width - 1), static_cast<uint16_t>(height - 1)},

                                  {static_cast<uint16_t>(width / 2), static_cast<uint16_t>(height / 2)},
                                  {static_cast<uint16_t>(width / 2), static_cast<uint16_t>(height / 2)},

                                  {0,                                0},
                                  {static_cast<uint16_t>(width - 1), static_cast<uint16_t>(height - 1)},

                                  {static_cast<uint16_t>(width - 1), static_cast<uint16_t>(height - 1)},
                                  {0,                                0}};
    ushort2 *d_arr;
    cudaMalloc(&d_arr, nrLines * 2 * sizeof(ushort2));
    cudaMemcpy(d_arr, input, nrLines * 2 * sizeof(ushort2), cudaMemcpyHostToDevice);

    // Call the function
    mapValidLineSegments(d_arr, nrLines, minLength, width, height);

    // Copy the result back to the host
    ushort2 result[nrLines * 2];
    cudaMemcpy(result, d_arr, nrLines * 2 * sizeof(ushort2), cudaMemcpyDeviceToHost);

    // Check the result
    for (uint32_t i = 0; i < nrLines; i++) {
        ushort2 oStart = input[2 * i];
        ushort2 oEnd = input[2 * i + 1];
        uint16_t oDistance = (oStart.x > oEnd.x ? oStart.x - oEnd.x : oEnd.x - oStart.x) +
                             (oStart.y > oEnd.y ? oStart.y - oEnd.y : oEnd.y - oStart.y);
        ushort2 start = result[2 * i];
        ushort2 end = result[2 * i + 1];
        uint16_t distance = (start.x > end.x ? start.x - end.x : end.x - start.x) +
                            (start.y > end.y ? start.y - end.y : end.y - start.y);

        ASSERT_LE(start.x, width);
        ASSERT_LE(start.y, height);
        ASSERT_LE(end.x, width);
        ASSERT_LE(end.y, height);

        if (oDistance >= minLength) {
            ASSERT_EQ(start.x, oStart.x);
            ASSERT_EQ(start.y, oStart.y);
            ASSERT_EQ(end.x, oEnd.x);
            ASSERT_EQ(end.y, oEnd.y);
        } else
            ASSERT_EQ(distance, minLength)
                                        << "Coordinates " << i << " ((" << start.x << ", " << start.y << "), (" << end.x
                                        << ", " << end.y
                                        << ")) with distance " << distance << " is shorter than minLength ("
                                        << minLength << ")";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}