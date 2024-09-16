#include <cmath>
#include <cuda_runtime.h>

#include "hardware_limits.hh"
#include "logger.hh"

static int max_threads_per_block_1d;
static int max_threads_per_block_2d;
static int max_blocks;

void init_hardware_limits()
{
    static bool initialized = false;

    if (!initialized)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_threads_per_block_1d = prop.maxThreadsPerBlock;
        max_threads_per_block_2d = static_cast<unsigned int>(sqrt(prop.maxThreadsPerBlock));
        max_blocks = prop.maxGridSize[0];
        LOG_INFO("Hardware limits: max threads per block 1D: {}, max threads per block 2D: {}, max blocks: {}",
            max_threads_per_block_1d, max_threads_per_block_2d, max_blocks);
        initialized = true;
    }
}

unsigned int get_max_threads_1d()
{
    init_hardware_limits();

    return max_threads_per_block_1d;
}

unsigned int get_max_threads_2d()
{
    init_hardware_limits();

    return max_threads_per_block_2d;
}

unsigned int get_max_blocks()
{
    init_hardware_limits();

    return max_blocks;
}