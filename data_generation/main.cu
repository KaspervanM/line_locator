#include <iostream>
#include <fstream>
#include <string>
#include "cudaFunctions.cu"

const int WIDTH = 28;
const int HEIGHT = 28;
const int NR_IMAGES = 20;

int main() {
    // Set a seed value (you can use any suitable method to generate a seed)
    unsigned long long seed = 1;

    int32_t nrPixels = WIDTH * HEIGHT * NR_IMAGES;
    auto *d_noisyImages = (uint8_t *) generateNoise(seed, nrPixels);

    // Map the range of the noise to [128, 255]
    mapRange(d_noisyImages, nrPixels, 128, 255);

    uint32_t coordinateValuesSize = NR_IMAGES * 2 * sizeof(ushort2) / sizeof(uint8_t);
    auto *d_coordinates = (ushort2 *) generateNoise(seed, coordinateValuesSize);

    // Filter the coordinates to be valid
    mapValidLineSegments(d_coordinates, NR_IMAGES, 5, WIDTH, HEIGHT);

    // add the line segments to the images
    uint8_t nrColors = 2;
    uint8_t h_colors[] = {0, 100};
    uint8_t nrWidths = 3;
    uint16_t h_widths[] = {1, 4, 2};
    uint8_t *d_colors;
    uint16_t *d_widths;
    cudaMalloc(&d_colors, nrColors * sizeof(uint8_t));
    cudaMalloc(&d_widths, nrWidths * sizeof(uint16_t));
    cudaMemcpy(d_colors, h_colors, nrColors * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_widths, h_widths, nrWidths * sizeof(uint16_t), cudaMemcpyHostToDevice);
    drawLine(d_noisyImages, NR_IMAGES, d_coordinates, WIDTH, HEIGHT, nrColors, d_colors, nrWidths, d_widths);

    // Allocate memory for the image on the host
    auto h_image = (uint8_t *) malloc(nrPixels * sizeof(uint8_t));
    cudaMemcpy(h_image, d_noisyImages, nrPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    auto h_coordinates = (ushort2 *) malloc(coordinateValuesSize);
    cudaMemcpy(h_coordinates, d_coordinates, coordinateValuesSize, cudaMemcpyDeviceToHost);

    // Print the coordinates
    for (int i = 0; i < NR_IMAGES; i++) {
        std::cout << h_coordinates[2 * i].x << " " << h_coordinates[2 * i].y << " "
                  << h_coordinates[2 * i + 1].x << " " << h_coordinates[2 * i + 1].y << std::endl;
    }

    // Save each image as a PGM file
    for (int i = 0; i < NR_IMAGES; i++) {
        std::ofstream pgmFile("output" + std::to_string(i) + ".pgm", std::ios::out | std::ios::binary);
        pgmFile << "P5\n" << WIDTH << " " << HEIGHT << "\n255\n";
        pgmFile.write(reinterpret_cast<char *>(h_image + i * WIDTH * HEIGHT), WIDTH * HEIGHT);
        pgmFile.close();
    }

    // Free host memory
    delete[] h_image;

    return 0;
}