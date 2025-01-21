#include "gtest/gtest.h"
#include "map.cuh"
#include "cuda_memory.cuh"
#include "test_disable_log.hh"

static constexpr cudaStream_t stream = 0;

enum class MAP_OPERATION
{
    DIVIDE = 0,
    MULTIPLY
};

template <typename T>
static bool check_result(const T* const h_expected, T* const d_got, const size_t size)
{
    T* h_got = new T[size];
    cudaXMemcpy(h_got, d_got, sizeof(T) * size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; i++)
    {
        if (h_expected[i] != h_got[i])
        {
            delete[] h_got;
            return false;
        }
    }
    delete[] h_got;
    return true;
}

// Meta programmation
template <typename T, MAP_OPERATION OP>
static void cpu_map(T* const input, const size_t size, T value)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (OP == MAP_OPERATION::DIVIDE)
            input[i] /= value;
        else if (OP == MAP_OPERATION::MULTIPLY)
            input[i] *= value;
    }
}

template <typename T, MAP_OPERATION OP>
static void map_test(size_t size, T value)
{
    T* h_data = new T[size];
    T* d_data;
    cudaXMalloc(&d_data, sizeof(T) * size);

    for (unsigned int i = 0; i < size; ++i)
    {
        const unsigned int tmp = i % 1000000;
        h_data[i] = static_cast<T>(tmp);
    }

    cudaXMemcpy(d_data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice);
    cpu_map<T, OP>(h_data, size, value);

    if (OP == MAP_OPERATION::DIVIDE)
        map_divide(d_data, d_data, size, value, stream);
    else if (OP == MAP_OPERATION::MULTIPLY)
        map_multiply(d_data, d_data, size, value, stream);
    cudaXStreamSynchronize(stream);

    ASSERT_TRUE(check_result(h_data, d_data, size));

    cudaXFree(d_data);
    delete[] h_data;
}

TEST(MapTest, MappNotDivisbleBy4Multiply)
{
    constexpr size_t size = 103;
    constexpr float value = 5.f;
    map_test<float, MAP_OPERATION::MULTIPLY>(size, value);
}

TEST(MapTest, MappDivisbleBy4Multiply)
{
    constexpr size_t size = 512 * 1024;
    constexpr float value = 5.f;
    map_test<float, MAP_OPERATION::MULTIPLY>(size, value);
}

TEST(MapTest, MappNotDivisbleBy4Divide)
{
    constexpr size_t size = 103;
    constexpr float value = 5.f;
    map_test<float, MAP_OPERATION::DIVIDE>(size, value);
}

TEST(MapTest, MappDivisbleBy4Divide)
{
    constexpr size_t size = 512 * 1024;
    constexpr float value = 5.f;
    map_test<float, MAP_OPERATION::DIVIDE>(size, value);
}
