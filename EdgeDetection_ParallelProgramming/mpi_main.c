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

uint8_t** split_image(uint8_t* input_image, int height, int width, int num_procs);
uint8_t* edgeDetection(uint8_t* input_image, int old_width, int new_width, int new_height);
uint8_t* combine_image(int chunk_number, int chunk_width, int chunk_height, uint8_t** edge_image_arr);

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (rank == 0) {
        int width, height, bpp;

        // Reading the image in grey colors
        uint8_t* input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);

        printf("Width: %d  Height: %d \n", width, height);
        printf("Input: %s , Output: %s  \n", argv[1], argv[2]);

        int chunk_height = height / num_procs;
        int new_width = (int)round(width);
        int new_height = (int)round(chunk_height);

        double start_time = MPI_Wtime(); //START TIMER

        // Allocating memory for downsized chunks
        uint8_t** edge_image_chunks = (uint8_t**)malloc(num_procs * sizeof(uint8_t*));

        // Split the image into horizontal chunks and save it into an 2d array
        uint8_t** image_chunks = split_image(input_image, height, num_procs, width);

        // Send image chunks and other variables to processes
        for (int i = 1; i < num_procs; i++) {
            MPI_Send(&chunk_height, sizeof(int), MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&width, sizeof(int), MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&new_width, sizeof(int), MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&new_height, sizeof(int), MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(image_chunks[i], chunk_height * width * sizeof(uint8_t), MPI_UINT8_T, i, 0, MPI_COMM_WORLD);
        }

        // Downsize the first part of the image on the root process and put it into array
        edge_image_chunks[0] = edgeDetection(image_chunks[0], width, new_width, new_height);

        // Receive the downsized image chunks from processors
        for (int i = 1; i < num_procs; i++) {
            uint8_t* edge_image_part = (uint8_t*)malloc(new_width * new_height * CHANNEL_NUM * sizeof(uint8_t));
            MPI_Recv(edge_image_part, new_height * new_width * sizeof(uint8_t), MPI_UINT8_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Append to downsized array for combining
            edge_image_chunks[i] = edge_image_part;
        }

        // Combine image chunks into one image
        uint8_t* output_image = combine_image(num_procs, new_width, new_height, edge_image_chunks);

        // Free the allocated memory
        for (int i = 0; i < num_procs; i++) {
            free(edge_image_chunks[i]);
        }
        free(edge_image_chunks);

        double end_time = MPI_Wtime(); //  END TIMER
        printf("(0) Elapsed time: %f seconds\n", end_time - start_time);

        // Save the output image
        stbi_write_jpg(argv[2], new_width, new_height * num_procs, CHANNEL_NUM, output_image, new_width * CHANNEL_NUM);

        // Free the allocated memory
        free(output_image);
    }
    else {
        int chunk_height, width, new_width, new_height;

        // Receive the necessary variables
        MPI_Recv(&chunk_height, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&width, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&new_width, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&new_height, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive the corresponding part of the image
        uint8_t* image_chunk = (uint8_t*)malloc(width * chunk_height * CHANNEL_NUM * sizeof(uint8_t));
        MPI_Recv(image_chunk, chunk_height * width * sizeof(uint8_t), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Downsize and send to main process
        MPI_Send(edgeDetection(image_chunk, width, new_width, new_height), new_width * new_height * sizeof(uint8_t), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();

    return 0;
}

uint8_t** split_image(uint8_t* input_image, int height, int num_procs, int width) {
    // Calculate the height of each chunk
    int chunk_height = height / num_procs;

    // Allocate memory for the array of chunk images
    uint8_t** chunk_images = (uint8_t**)malloc(num_procs * sizeof(uint8_t*));

    // Loop over the number of chunks and create a smaller image for each chunk
    for (int i = 0; i < num_procs; i++) {
        // Calculate the y-coordinate of the top of the chunk
        int y = i * chunk_height;

        // Store the chunk image in the array
        chunk_images[i] = input_image + y * width * CHANNEL_NUM, width* chunk_height* CHANNEL_NUM;
    }

    return chunk_images;
}

uint8_t* edgeDetection(uint8_t* input_image, int old_width, int new_width, int new_height) {
    uint8_t* output_image = (uint8_t*)malloc(new_width * new_height * CHANNEL_NUM * sizeof(uint8_t));
    // Sobel operators
    int sobel_x[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobel_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Temporary buffers for gradient values
    int* gradient_x = (int*)malloc(new_height * new_width * sizeof(int));
    int* gradient_y = (int*)malloc(new_height * new_width * sizeof(int));

    // Compute gradient in x direction
    for (int i = 1; i < new_height - 1; i++) {
        for (int j = 1; j < new_width - 1; j++) {
            int sum_x = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum_x += input_image[(i + k) * new_width + (j + l)] * sobel_x[k + 1][l + 1];
                }
            }
            gradient_x[i * new_width + j] = sum_x;
        }
    }

    // Compute gradient in y direction
    for (int i = 1; i < new_height - 1; i++) {
        for (int j = 1; j < new_width - 1; j++) {
            int sum_y = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum_y += input_image[(i + k) * new_width + (j + l)] * sobel_y[k + 1][l + 1];
                }
            }
            gradient_y[i * new_width + j] = sum_y;
        }
    }

    // Compute gradient magnitude
    for (int i = 1; i < new_height - 1; i++) {
        for (int j = 1; j < new_width - 1; j++) {
            int gx = gradient_x[i * new_width + j];
            int gy = gradient_y[i * new_width + j];
            output_image[i * new_width + j] = (uint8_t)sqrt(gx * gx + gy * gy);
        }
    }

    // Free allocated memory
    free(gradient_x);
    free(gradient_y);

    return output_image;

}


uint8_t* combine_image(int chunks, int chunk_width, int chunk_height, uint8_t** edge_image_arr) {
    int whole_width = chunk_width;
    int whole_height = chunk_height * chunks;

    // Allocate memory for the whole image
    uint8_t* whole_image = (uint8_t*)malloc(whole_width * whole_height * CHANNEL_NUM * sizeof(uint8_t));

    // Loop over all the chunks and copy their pixel data into the whole image array
    for (int i = 0; i < chunks; i++) {
        for (int j = 0; j < chunk_height; j++) {
            uint8_t* chunk_row = edge_image_arr[i] + j * chunk_width * CHANNEL_NUM;
            uint8_t* whole_row = whole_image + (i * chunk_height + j) * whole_width * CHANNEL_NUM;
            memcpy(whole_row, chunk_row, chunk_width * CHANNEL_NUM);
        }
    }

    return whole_image;
}
