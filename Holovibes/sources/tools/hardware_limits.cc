#include <cmath>
#include <cuda_runtime.h>

#include "hardware_limits.hh"

unsigned int get_max_threads_1d()
{
    static int max_threads_per_block_1d;
    static bool initialized = false;

    if (!initialized)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_threads_per_block_1d = prop.maxThreadsPerBlock;
        initialized = true;
    }

    return max_threads_per_block_1d;
}

unsigned int get_max_threads_2d()
{
    static int max_threads_per_block_2d;
    static bool initialized = false;

    if (!initialized)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_threads_per_block_2d =
            static_cast<unsigned int>(sqrt(prop.maxThreadsPerBlock));
        initialized = true;
    }

    return max_threads_per_block_2d;
}

unsigned int get_max_blocks()
{
    static int max_blocks;
    static bool initialized = false;

    if (!initialized)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_blocks = prop.maxGridSize[0];
        initialized = true;
    }

    return max_blocks;
}