#include <curand.h>
#include <cstdint>

void *generateNoise(uint64_t seed, uint32_t n) {
    // Generate the grayscale image with noise
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // Generate the grayscale image with noise
    uint32_t *d_out;
    uint32_t n32 = n * sizeof(uint8_t) / sizeof(uint32_t) + (n & 3);
    cudaMalloc(&d_out, n32 * sizeof(uint32_t));
    curandGenerate(gen, (uint32_t *) d_out, n32);

    // Free cuRAND states
    curandDestroyGenerator(gen);

    return d_out;
}

__global__ void kern_mapRange(int32_t n, uint8_t *arr, uint8_t min, uint8_t max) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = (arr[i] % (max - min)) + min;
}

// minLength is in Manhattan distance
// randCoords are already mapped to be between valid width and height
__global__ void
kern_filterValidLineSegments(int32_t n, ushort2 *randCoords, uint16_t minLength, uint16_t width, uint16_t height) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    randCoords[2 * i].x %= width;
    randCoords[2 * i].y %= height;
    randCoords[2 * i + 1].x %= width;
    randCoords[2 * i + 1].y %= height;

    uint16_t distance = (randCoords[2 * i].x > randCoords[2 * i + 1].x ? randCoords[2 * i].x - randCoords[2 * i + 1].x :
                         randCoords[2 * i + 1].x - randCoords[2 * i].x)
                        + (randCoords[2 * i].y > randCoords[2 * i + 1].y ? randCoords[2 * i].y - randCoords[2 * i + 1].y
                                                                         : randCoords[2 * i + 1].y -
                                                                           randCoords[2 * i].y);
    int diff = minLength - distance;
    while (diff > 0) {
        if (randCoords[2 * i].x > 0) {
            randCoords[2 * i].x -= 1;
        } else if (randCoords[2 * i].x > 0) {
            randCoords[2 * i].y -= 1;
        } else if (randCoords[2 * i + 1].x < width - 1) {
            randCoords[2 * i + 1].x += 1;
        } else if (randCoords[2 * i + 1].y < height - 1) {
            randCoords[2 * i + 1].y += 1;
        } else {
            break;
        }
        distance = (randCoords[2 * i].x > randCoords[2 * i + 1].x ? randCoords[2 * i].x - randCoords[2 * i + 1].x :
                    randCoords[2 * i + 1].x - randCoords[2 * i].x)
                   + (randCoords[2 * i].y > randCoords[2 * i + 1].y ? randCoords[2 * i].y - randCoords[2 * i + 1].y :
                      randCoords[2 * i + 1].y - randCoords[2 * i].y);
        diff = minLength - distance;
    }
}

__device__ void
drawCircle(uint8_t *arr, uint32_t imageOffset, uint16_t imageWidth, uint16_t imageHeight, uint16_t x, uint16_t y,
           uint8_t color, uint8_t radius) {
    radius -= 1;
    uint16_t radiusSquared = radius * radius;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            uint16_t distanceSquared = i * i + j * j;
            if (distanceSquared <= radiusSquared) {
                uint16_t xCoord = x + i;
                uint16_t yCoord = y + j;
                if (xCoord < imageWidth && yCoord < imageHeight) {
                    arr[imageOffset + yCoord * imageWidth + xCoord] = color;
                }
            }
        }
    }
}

// x as percentage of gradient [0, 100]
template<typename T>
__device__ T interpolateGradient(uint32_t nrVals, T *arr, uint8_t x) {
    if (nrVals == 1) return arr[0];

    uint32_t first_index = ((nrVals - 1) * x / 100);
    uint32_t second_index = first_index + 1;
    if (second_index > nrVals - 1)
        second_index = nrVals - 2;

    T first = arr[first_index];
    T second = arr[second_index];
    float between = __uint2float_rn((nrVals - 1) * x) / 100.f - __uint2float_rn(first_index);
    return first * (1 - between) + second * between;
}

// generates a line segment between two points with varying width and color
__global__ void
kern_drawLine(int32_t n, uint8_t *arr, ushort2 *coords, uint16_t imageWidth, uint16_t imageHeight, uint8_t nrColors,
              uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t imageOffset = i * imageWidth * imageHeight;

    ushort2 start = coords[2 * i];
    ushort2 end = coords[2 * i + 1];

    uint16_t xDiff = start.x > end.x ? start.x - end.x : end.x - start.x;
    uint16_t yDiff = start.y > end.y ? start.y - end.y : end.y - start.y;

    // draw line on current image by drawing a circle at each point of certain width and color
    uint16_t maxAbsDiff = max(xDiff, yDiff);
    uint16_t step = maxAbsDiff + 1;
    for (int j = 0; j < step; j++) {
        uint16_t x = start.x + xDiff * j / step;
        uint16_t y = start.y + yDiff * j / step;
        uint8_t color = __float2int_rn(interpolateGradient(nrColors, colors, j * 100 / step));
        uint16_t width = __float2int_rn(interpolateGradient(nrWidths, widths, j * 100 / step));
        drawCircle(arr, imageOffset, imageWidth, imageHeight, x, y, color, width);
    }
}


void mapRange(uint8_t *arr, int32_t n, uint8_t min, uint8_t max) {
    int32_t blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_mapRange, 0, 0);
    gridSize = (n + blockSize - 1) / blockSize;
    kern_mapRange<<< gridSize, blockSize >>>(n, arr, min, max);
    cudaDeviceSynchronize();
}

void mapValidLineSegments(ushort2 *randCoords, int32_t n, uint16_t minLength, uint16_t width, uint16_t height) {
    int32_t blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_filterValidLineSegments, 0, 0);
    gridSize = (n + blockSize - 1) / blockSize;
    kern_filterValidLineSegments<<< gridSize, blockSize >>>(n, randCoords, minLength, width, height);
    cudaDeviceSynchronize();
}

void drawLine(uint8_t *arr, int32_t n, ushort2 *coords, uint16_t width, uint16_t height, uint8_t nrColors,
              uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    int32_t blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_drawLine, 0, 0);
    gridSize = (n + blockSize - 1) / blockSize;
    kern_drawLine<<< gridSize, blockSize >>>(n, arr, coords, width, height, nrColors, colors, nrWidths, widths);
    cudaDeviceSynchronize();
}