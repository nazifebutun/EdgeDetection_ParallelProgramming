/**
 *
 * CENG342 Project-1
 *
 * Edge Detection
 *
 * Usage:  main <input.jpg> <output.jpg>
 *
 * @group_id 00
 * @author  your names
 *
 * @version 1.0, 02 March 2024
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

 //Do not use global variables

void seq_edgeDetection(uint8_t* input_image, int width, int height);


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int width, height, bpp;

    // Reading the image in grey colors
    uint8_t* input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);

    printf("Width: %d  Height: %d \n", width, height);
    printf("Input: %s , Output: %s  \n", argv[1], argv[2]);

    // start the timer
    double time1 = MPI_Wtime();

    seq_edgeDetection(input_image, width, height);
    double time2 = MPI_Wtime();
    printf("Elapsed time: %lf \n", time2 - time1);


    // Storing the image 
    stbi_write_jpg(argv[2], width, height, CHANNEL_NUM, input_image, 100);
    stbi_image_free(input_image);

    MPI_Finalize();
    return 0;
}

void seq_edgeDetection(uint8_t* local_image, int local_width, int local_height)
{
    // Sobel operators
    int sobel_x[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobel_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Temporary buffers for gradient values
    int* gradient_x = (int*)malloc(local_height * local_width * sizeof(int));
    int* gradient_y = (int*)malloc(local_height * local_width * sizeof(int));

    // Compute gradient in x direction
    for (int i = 1; i < local_height - 1; i++) {
        for (int j = 1; j < local_width - 1; j++) {
            int sum_x = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum_x += local_image[(i + k) * local_width + (j + l)] * sobel_x[k + 1][l + 1];
                }
            }
            gradient_x[i * local_width + j] = sum_x;
        }
    }

    // Compute gradient in y direction
    for (int i = 1; i < local_height - 1; i++) {
        for (int j = 1; j < local_width - 1; j++) {
            int sum_y = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum_y += local_image[(i + k) * local_width + (j + l)] * sobel_y[k + 1][l + 1];
                }
            }
            gradient_y[i * local_width + j] = sum_y;
        }
    }

    // Compute gradient magnitude
    for (int i = 1; i < local_height - 1; i++) {
        for (int j = 1; j < local_width - 1; j++) {
            int gx = gradient_x[i * local_width + j];
            int gy = gradient_y[i * local_width + j];
            local_image[i * local_width + j] = (uint8_t)sqrt(gx * gx + gy * gy);
        }
    }

    // Free allocated memory
    free(gradient_x);
    free(gradient_y);
}