#include "gtest/gtest.h"

// Include headers where comp_dgaussian is declared.
// Make sure that comp_dgaussian is declared in one of these headers.
#include "cuda_memory.cuh"
#include "vesselness_filter.cuh" // or "tools_analysis.cuh", if the function is declared there
#include "batch_input_queue.hh"
#include "tools_analysis.cuh"

#include <array>
#include <cuda_runtime.h>
#include <algorithm>

// Test for comp_dgaussian when the derivative order n equals 2.
TEST(AnalysisTest, comp_dgaussian_n2)
{
    // Define the host input array.
    constexpr float h_input[3] = {-1.0f, 0.0f, 1.0f};
    constexpr float sigma = 0.1f;
    constexpr int n = 2;
    constexpr size_t len = 3; // Number of elements in the array

    // Allocate device memory for input and output arrays.
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, len * sizeof(float));
    cudaMalloc(&d_output, len * sizeof(float));

    // Copy the host input array to device memory.
    cudaMemcpy(d_input, h_input, len * sizeof(float), cudaMemcpyHostToDevice);

    // Call the host wrapper comp_dgaussian.
    // The function uses a CUDA stream; here we pass 0 (the default stream).
    comp_dgaussian(d_output, d_input, len, sigma, n, 0);

    // Allocate a host array to receive the computed results.
    std::array<float, 3> h_output;
    cudaMemcpy(h_output.data(), d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    // Define the expected output values.
    // (These values are taken from your provided test expectations.)
    std::array<float, 3> expected = {
        7.6177e-18f, // Expected value for input -1
        -398.9423f,  // Expected value for input  0
        7.6177e-18f  // Expected value for input  1
    };

    // Compare each computed value with the expected value using ASSERT_NEAR.
    // A tolerance of 1e-5 is used (adjust as needed).
    for (size_t i = 0; i < len; i++)
    {
        ASSERT_NEAR(h_output[i], expected[i], 1e-5f);
    }

    // Free the allocated device memory.
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test for comp_dgaussian when the derivative order n equals 0.
TEST(AnalysisTest, comp_dgaussian_n0)
{
    // Define the host input array.
    constexpr float h_input[3] = {-1.0f, 0.0f, 1.0f};
    constexpr float sigma = 0.1f;
    constexpr int n = 0;
    constexpr size_t len = 3; // Number of elements in the array

    // Allocate device memory for input and output arrays.
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, len * sizeof(float));
    cudaMalloc(&d_output, len * sizeof(float));

    // Copy the host input array to device memory.
    cudaMemcpy(d_input, h_input, len * sizeof(float), cudaMemcpyHostToDevice);

    // Call the host wrapper comp_dgaussian with stream = 0.
    comp_dgaussian(d_output, d_input, len, sigma, n, 0);

    // Allocate a host array to receive the computed results.
    std::array<float, 3> h_output;
    cudaMemcpy(h_output.data(), d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    // Define the expected output values.
    std::array<float, 3> expected = {
        7.6946e-22f, // Expected value for input -1
        3.9894f,     // Expected value for input  0
        7.6946e-22f  // Expected value for input  1
    };

    // Compare each computed value with the expected value using ASSERT_NEAR.
    for (size_t i = 0; i < len; i++)
    {
        ASSERT_NEAR(h_output[i], expected[i], 1e-5f);
    }

    // Free the allocated device memory.
    cudaFree(d_input);
    cudaFree(d_output);
}
