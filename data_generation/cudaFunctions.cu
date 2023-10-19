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

__device__ void
drawLineSegment(uint8_t *arr, uint32_t imageOffset, uint16_t imageWidth, uint16_t imageHeight, float x0, float y0,
                float x1, float y1, uint8_t color) {
    for (int x = 0; x < imageWidth; x++) {
        for (int y = 0; y < imageHeight; y++) {
            float AB = sqrtf((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            float AP = sqrtf((x - x0) * (x - x0) + (y - y0) * (y - y0));
            float PB = sqrtf((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));

            // adjust threshold to make the line thicker
            const float threshold = 2.f;
            if (fabs(AB - (AP + PB)) <= threshold) {
                arr[imageOffset + y * imageWidth + x] = color;
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
kern_drawLine(int32_t n, uint8_t *arr, ushort2 *coords, uint16_t imageWidth, uint16_t imageHeight,
              uint8_t nrColors,
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

__global__ void
kern_drawLine2(int32_t n, uint8_t *arr, ushort2 *coords, uint16_t imageWidth, uint16_t imageHeight,
               uint8_t nrColors,
               uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t imageOffset = i * imageWidth * imageHeight;

    float2 left_coordinate{static_cast<float>(coords[2 * i].x), static_cast<float>(coords[2 * i].y)};
    float2 right_coordinate{static_cast<float>(coords[2 * i + 1].x), static_cast<float>(coords[2 * i + 1].y)};

    // Calculate the vector representing the line
    float2 line_vector{right_coordinate.x - left_coordinate.x, right_coordinate.y - left_coordinate.y};
    float line_length = sqrtf(line_vector.x * line_vector.x + line_vector.y * line_vector.y);
    float2 line_direction{line_vector.x / line_length, line_vector.y / line_length};

    const int segmentsPerPixel = 8;

    // Determine the number of segments for the line
    int numSegments = static_cast<int>(roundf(ceilf(line_length)) * segmentsPerPixel);

    int32_t blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_mapRange, 0, 0);

    int blockSizeSqrt = __float2int_rn(sqrtf(float(blockSize)));

    const dim3 lineBlockDim(blockSizeSqrt, blockSizeSqrt);
    const dim3 lineGridDim((imageWidth + blockSizeSqrt - 1) / blockSizeSqrt,
                           (imageHeight + blockSizeSqrt) / blockSizeSqrt);

    for (int j = 0; j < numSegments; j++) {
        // Calculate the position along the line
        float t = static_cast<float>(j) / static_cast<float>(numSegments - 1);
        float2 position{left_coordinate.x + t * line_vector.x, left_coordinate.y + t * line_vector.y};

        // Interpolate grayscale and width values
        uint8_t color = __float2int_rn(interpolateGradient(nrColors, colors, j * 100 / (numSegments - 1)));
        float width = interpolateGradient(nrWidths, widths, j * 100 / (numSegments - 1));
        width = max(1.f, width);

        // Calculate the perpendicular vectors for the line segment
        float2 perpendicular_vector{-line_direction.y, line_direction.x};

        // Calculate the start and end points for the line segment
        float2 start_point{position.x - 0.5f * width * perpendicular_vector.x,
                           position.y - 0.5f * width * perpendicular_vector.y};
        float2 end_point{position.x + 0.5f * width * perpendicular_vector.x,
                         position.y + 0.5f * width * perpendicular_vector.y};

        drawLineSegment(arr, imageOffset, imageWidth, imageHeight, start_point.x,
                        start_point.y, end_point.x, end_point.y, color);
    }
}

__device__ int
det2x2(int2 a, int2 b) {
    return a.x * b.y - b.x * a.y;
}

__global__ void
kern_drawLine3(uint32_t n, uint8_t *arr, uint16_t imageWidth, uint16_t imageHeight, ushort2 *coords, uint8_t nrColors,
               uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= imageWidth * imageHeight) return;

    uint32_t imageOffset = n * imageWidth * imageHeight;

    uint16_t x = i % imageWidth;
    uint16_t y = i / imageWidth;

    int2 p0{static_cast<int>(coords[2 * n].x), static_cast<int>(coords[2 * n].y)};
    int2 p1{static_cast<int>(coords[2 * n + 1].x), static_cast<int>(coords[2 * n + 1].y)};

    int2 lineVector{p1.x - p0.x, p1.y - p0.y};
    int2 minuslineVector{p0.x - p1.x, p0.y - p1.y};
    int2 orthogonalVector{lineVector.y, -lineVector.x};
    auto D = float(det2x2(minuslineVector, orthogonalVector));
    int2 p0minusp{p0.x - x, p0.y - y};
    auto Dm = float(det2x2(p0minusp, orthogonalVector));
    float m = Dm / D;

    if (m < 0 || m > 1) return;

    // Interpolate grayscale and width values
    uint8_t color = __float2int_rn(interpolateGradient(nrColors, colors, m * 100));
    float width = interpolateGradient(nrWidths, widths, m * 100);
    width = max(1.f, width);

    float AB = sqrtf(float((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y)));
    float AP = sqrtf(float((x - p0.x) * (x - p0.x) + (y - p0.y) * (y - p0.y)));
    float PB = sqrtf(float((p1.x - x) * (p1.x - x) + (p1.y - y) * (p1.y - y)));

    // adjust threshold to make the line thicker
    const float threshold = width / 2.f;
    if (fabs(AB - (AP + PB)) <= threshold) {
        arr[imageOffset + y * imageWidth + x] = color;
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

void drawLine2(uint8_t *arr, int32_t n, ushort2 *coords, uint16_t width, uint16_t height, uint8_t nrColors,
               uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    int32_t blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_drawLine2, 0, 0);
    gridSize = (n + blockSize - 1) / blockSize;
    kern_drawLine2<<< gridSize, blockSize >>>(n, arr, coords, width, height, nrColors, colors, nrWidths, widths);
    cudaDeviceSynchronize();
}

void drawLine3(uint8_t *arr, int32_t n, ushort2 *coords, uint16_t width, uint16_t height, uint8_t nrColors,
               uint8_t *colors, uint8_t nrWidths, uint16_t *widths) {
    int32_t blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kern_drawLine3, 0, 0);
    gridSize = (n + blockSize - 1) / blockSize;
    for (int i = 0; i < n; i++) {
        kern_drawLine3<<< gridSize, blockSize >>>(i, arr, width, height, coords, nrColors, colors, nrWidths, widths);
    }
    cudaDeviceSynchronize();
}

