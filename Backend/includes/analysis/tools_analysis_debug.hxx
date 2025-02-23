#pragma once

#include <string>
#include "cuda_memory.cuh"
#include <cuda_runtime.h>

using uint = unsigned int;

template <typename T>
void write_1D_array_to_file(const T* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
                outFile << " "; // Separate values in a row by a space
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

template <typename T>
void print_in_file_gpu(const T* input, uint rows, uint col, std::string filename, cudaStream_t stream)
{
    if (input == nullptr)
        return;
    T* result = new T[rows * col];
    cudaXMemcpyAsync(result, input, rows * col * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    write_1D_array_to_file<T>(result, rows, col, "test_" + filename + ".txt");
}