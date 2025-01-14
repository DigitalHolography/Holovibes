#include "gtest/gtest.h"
#include "frame_reshape.cuh"
#include "cuda_memory.cuh"
#include "test_disable_log.hh"

TEST(SubSampleFrame, subsample_8x8_step_2)
{
    char* input = nullptr;
    char* output = nullptr;

    cudaXMalloc(&input, 8 * 8);
    cudaXMalloc(&output, 4 * 4);

    cudaXMemset(input, 0, 8 * 8);
    cudaXMemset(output, 0, 4 * 4);

    // clang-format off
    char input_image[8 * 8] = {
          1,  2,  3,  4,  5,  6,  7,  8,
          9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24,
         25, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47, 48,
         49, 50, 51, 52, 53, 54, 55, 56,
         57, 58, 59, 60, 61, 62, 63, 64
    };
    char ref_output_image[4 * 4] = {
         1, 11,  5, 15,
        18, 28, 22, 32,
        33, 43, 37, 47,
        50, 60, 54, 64
    };
    char output_image[4 * 4];
    // clang-format on

    cudaXMemcpy(input, input_image, 8 * 8, cudaMemcpyHostToDevice);
    subsample_frame(input, 8, 8, output, 2, sizeof(char), 0);
    cudaMemcpy(output_image, output, 4 * 4, cudaMemcpyDeviceToHost);

    ASSERT_EQ(memcmp(output_image, ref_output_image, 4 * 4), 0);
}

TEST(SubSampleFrame, subsample_8x8_step_4)
{
    char* input = nullptr;
    char* output = nullptr;

    cudaXMalloc(&input, 8 * 8);
    cudaXMalloc(&output, 2 * 2);

    cudaXMemset(input, 0, 8 * 8);
    cudaXMemset(output, 0, 2 * 2);

    // clang-format off
    char input_image[8 * 8] = {
          1,  2,  3,  4,  5,  6,  7,  8,
          9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24,
         25, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47, 48,
         49, 50, 51, 52, 53, 54, 55, 56,
         57, 58, 59, 60, 61, 62, 63, 64
    };
    char ref_output_image[2 * 2] = {
         1, 21,
         35, 55
    };
    char output_image[2 * 2];
    // clang-format on

    cudaXMemcpy(input, input_image, 8 * 8, cudaMemcpyHostToDevice);
    subsample_frame(input, 8, 8, output, 4, sizeof(char), 0);
    cudaMemcpy(output_image, output, 2 * 2, cudaMemcpyDeviceToHost);

    ASSERT_EQ(memcmp(output_image, ref_output_image, 2 * 2), 0);
}
