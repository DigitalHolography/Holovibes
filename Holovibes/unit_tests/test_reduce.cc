/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "cuda_memory.cuh"
#include "gtest/gtest.h"
#include "reduce.cuh"

using uint = unsigned int;

static double init_data(float** d_data, double** d_result, size_t size)
{
    float* h_data = new float[size];
    cudaXMalloc((void**)d_data, sizeof(float) * size);
    cudaXMalloc((void**)d_result, sizeof(double));
    cudaXMemset(*d_result, 0, sizeof(double));

    double result = 0.0;
    for (unsigned int i = 0; i < size; ++i)
    {
        unsigned int tmp = i % 10;
        h_data[i] = static_cast<float>(tmp);
        result += static_cast<double>(h_data[i]);
    }

    cudaXMemcpy(*d_data, h_data, sizeof(float) * size, cudaMemcpyHostToDevice);

    delete h_data;

    return result;
}

static void check_result(double* d_result, double expected, float* d_data)
{
    double h_result;
    cudaXMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaXFree(d_result);
    cudaXFree(d_data);

    ASSERT_EQ(expected, h_result);
}

TEST(ReduceTest, RegularTest)
{
    const uint image_width = 1024;
    const uint image_height = 1024;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, AnamorphicTest)
{
    const uint image_width = 1024;
    const uint image_height = 512;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, OddTest)
{
    const uint image_width = 1024 + 1;
    const uint image_height = 1024 + 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_32x32)
{
    const uint image_width = 32;
    const uint image_height = 32;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_31x31)
{
    const uint image_width = 31;
    const uint image_height = 31;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_33x32)
{
    const uint image_width = 33;
    const uint image_height = 33;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x128)
{
    const uint image_width = 1024;
    const uint image_height = 128;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x127)
{
    const uint image_width = 1024;
    const uint image_height = 127;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x129)
{
    const uint image_width = 1024;
    const uint image_height = 129;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1023x129)
{
    const uint image_width = 1024;
    const uint image_height = 129;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_64x64)
{
    const uint image_width = 64;
    const uint image_height = 64;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_not_power_2)
{
    const uint image_width = 120;
    const uint image_height = 120;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_127)
{
    const uint image_width = 127;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_128)
{
    const uint image_width = 128;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_129)
{
    const uint image_width = 129;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_63)
{
    const uint image_width = 63;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_64)
{
    const uint image_width = 64;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_65)
{
    const uint image_width = 65;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_29)
{
    const uint image_width = 29;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_2048x2048)
{
    const uint image_width = 2048;
    const uint image_height = 2048;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_256x2048)
{
    const uint image_width = 256;
    const uint image_height = 2048;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_256x1024)
{
    const uint image_width = 256;
    const uint image_height = 1024;

    float* d_data;
    double* d_result;
    double expected = init_data(&d_data, &d_result, image_height * image_width);

    gpu_reduce(d_data, d_result, image_width * image_height);
    cudaDeviceSynchronize();

    check_result(d_result, expected, d_data);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}