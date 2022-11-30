#include <algorithm>

#include "cuda_memory.cuh"
#include "gtest/gtest.h"
#include "test_reduce.cuh"

using namespace holovibes;

static constexpr cudaStream_t stream = 0;

template <typename T, typename U>
static U init_data_sum(T** d_data, U** d_result, size_t size)
{
    T* h_data = new T[size];
    cudaXMalloc((void**)d_data, sizeof(T) * size);
    cudaXMalloc((void**)d_result, sizeof(U));
    cudaXMemset(*d_result, 0, sizeof(U));

    U result = 0.0;
    for (unsigned int i = 0; i < size; ++i)
    {
        const uint tmp = i % 10;
        h_data[i] = static_cast<T>(tmp);
        result += static_cast<U>(h_data[i]);
    }

    cudaXMemcpy(*d_data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice);

    delete[] h_data;

    return result;
}

template <typename T>
static T init_data_min(T** d_data, T** d_result, size_t size)
{
    T* h_data = new T[size];
    cudaXMalloc((void**)d_data, sizeof(T) * size);
    cudaXMalloc((void**)d_result, sizeof(T));
    cudaXMemset(*d_result, 0, sizeof(T));

    T result = 1;
    for (unsigned int i = 0; i < size; ++i)
    {
        h_data[i] = static_cast<T>((i % 1000) + 1); // >= 1
        result = std::min(result, h_data[i]);
    }

    cudaXMemcpy(*d_data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice);

    delete[] h_data;

    return result;
}

template <typename T>
static T init_data_max(T** d_data, T** d_result, size_t size)
{
    T* h_data = new T[size];
    cudaXMalloc((void**)d_data, sizeof(T) * size);
    cudaXMalloc((void**)d_result, sizeof(T));
    cudaXMemset(*d_result, 0, sizeof(T));

    T result = static_cast<T>(-99999);
    for (unsigned int i = 0; i < size; ++i)
    {
        h_data[i] = -static_cast<T>(((i % 1000) + 1)); // <= -1;
        result = std::max(result, h_data[i]);
    }

    cudaXMemcpy(*d_data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice);

    delete[] h_data;

    return result;
}

template <typename T, typename U>
static void check_result(U* d_result, U expected, T* d_data)
{
    U h_result;
    cudaXMemcpy(&h_result, d_result, sizeof(U), cudaMemcpyDeviceToHost);

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
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, AnamorphicTest)
{
    const uint image_width = 1024;
    const uint image_height = 512;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, OddTest)
{
    const uint image_width = 1024 + 1;
    const uint image_height = 1024 + 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_32x32)
{
    const uint image_width = 32;
    const uint image_height = 32;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_31x31)
{
    const uint image_width = 31;
    const uint image_height = 31;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_33x32)
{
    const uint image_width = 33;
    const uint image_height = 33;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x128)
{
    const uint image_width = 1024;
    const uint image_height = 128;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x127)
{
    const uint image_width = 1024;
    const uint image_height = 127;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1024x129)
{
    const uint image_width = 1024;
    const uint image_height = 129;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_1023x129)
{
    const uint image_width = 1024;
    const uint image_height = 129;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_64x64)
{
    const uint image_width = 64;
    const uint image_height = 64;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_not_power_2)
{
    const uint image_width = 120;
    const uint image_height = 120;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_127)
{
    const uint image_width = 127;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_128)
{
    const uint image_width = 128;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_129)
{
    const uint image_width = 129;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_63)
{
    const uint image_width = 63;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_64)
{
    const uint image_width = 64;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_65)
{
    const uint image_width = 65;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_29)
{
    const uint image_width = 29;
    const uint image_height = 1;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_2048x2048)
{
    const uint image_width = 2048;
    const uint image_height = 2048;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_256x2048)
{
    const uint image_width = 256;
    const uint image_height = 2048;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_256x1024)
{
    const uint image_width = 256;
    const uint image_height = 1024;

    float* d_data;
    double* d_result;
    double expected = init_data_sum(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_add(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_min_1024x1024)
{
    const uint image_width = 1024;
    const uint image_height = 1024;

    double* d_data;
    double* d_result;
    double expected = init_data_min(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_min(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_int_max_1024x1024)
{
    const uint image_width = 1024;
    const uint image_height = 1024;

    int* d_data;
    int* d_result;
    int expected = init_data_max(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_max(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}

TEST(ReduceTest, Test_float_max_1024x1024)
{
    const uint image_width = 1024;
    const uint image_height = 1024;

    float* d_data;
    float* d_result;
    float expected = init_data_max(&d_data, &d_result, image_height * image_width);

    test_gpu_reduce_max(d_data, d_result, image_width * image_height);
    cudaXStreamSynchronize(stream);

    check_result(d_result, expected, d_data);
}
