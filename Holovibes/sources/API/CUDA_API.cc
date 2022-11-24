
#include "API.hh"
#include "holovibes_config.hh"
#include "tools.cuh"
#include "cuda_memory.cuh"
#include "CUDA_API.hh"

namespace holovibes::api
{
void check_cuda_graphic_card()
{
    std::string error_message;
    int device;
    int nDevices;
    int min_compute_capability = 35;
    int max_compute_capability = 86;
    int compute_capability;
    cudaError_t status;
    cudaDeviceProp props;

    /* Checking for Compute Capability */
    if ((status = cudaGetDeviceCount(&nDevices)) == cudaSuccess)
    {
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);

        compute_capability = props.major * 10 + props.minor;

        if (compute_capability >= min_compute_capability && compute_capability <= max_compute_capability)
            return;
        else
            error_message = "CUDA graphic card not supported.\n";
    }
    else
        error_message = "No CUDA graphic card detected.\n"
                        "You will not be able to run Holovibes.\n\n"
                        "Try to update your graphic drivers.";

    throw std::runtime_error(error_message);
}
} // namespace holovibes::api
