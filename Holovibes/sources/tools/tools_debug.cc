#include "tools_debug.hh"

#include <iostream>
#include <cufftXt.h>
#include <type_traits>

#include "cuda_memory.cuh"
#include "logger.hh"

namespace holovibes
{
template <typename T>
void device_print(T* d_data, size_t offset, size_t nb_elts)
{
    // Allocate host memory
    T* h_data = (T*)malloc(sizeof(T) * nb_elts);
    cudaXMemcpy(h_data, d_data + offset, sizeof(T) * nb_elts, cudaMemcpyDeviceToHost);

    // Use if-else or specialized template functions for custom printing logic
    if constexpr (std::is_same<T, cuComplex>::value)
    {
        for (size_t i = 0; i < nb_elts; ++i)
        {
            std::cout << h_data[i].x << " " << h_data[i].y << "| ";
        }
    }
    else
    {
        for (size_t i = 0; i < nb_elts; ++i)
        {
            std::cout << static_cast<uint>(h_data[i]) << " ";
        }
    }

    std::cout << std::endl; // New line for better readability

    free(h_data);
}

// Specialization for cuComplex to handle its unique printing logic
template <>
void device_print<cuComplex>(cuComplex* d_data, size_t offset, size_t nb_elts)
{
    cuComplex* h_data = (cuComplex*)malloc(sizeof(cuComplex) * nb_elts);
    cudaXMemcpy(h_data, d_data + offset, sizeof(cuComplex) * nb_elts, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < nb_elts; ++i)
    {
        std::cout << h_data[i].x << " " << h_data[i].y << "| ";
    }
    std::cout << std::endl; // New line for better readability
    free(h_data);
}

} // namespace holovibes
