#include "gpu_stats.hh"

#include <sstream>

#include "tools.hh"

namespace holovibes
{
const std::string get_load_color(float load, float max_load, float orange_ratio, float red_ratio)
{
    const float ratio = (load / max_load);
    if (ratio < orange_ratio)
        return "white";
    if (ratio < red_ratio)
        return "orange";
    return "red";
}

int get_gpu_load(nvmlUtilization_t* gpuLoad)
{
    nvmlDevice_t device;

    // Initialize NVML
    if (nvmlInit() != NVML_SUCCESS)
        return -1;

    // Get the device handle (assuming only one GPU is present)
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS)
    {
        nvmlShutdown();
        return -1;
    }

    // Query GPU load
    if (nvmlDeviceGetUtilizationRates(device, gpuLoad) != NVML_SUCCESS)
    {
        nvmlShutdown();
        return -1;
    }

    // Shutdown NVML
    return nvmlShutdown();
}

std::string gpu_load()
{
    nvmlUtilization_t gpuLoad;
    std::stringstream ss;
    ss << "<td>GPU load</td>";

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU load
    float load = static_cast<float>(gpuLoad.gpu);
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string gpu_load_as_number()
{
    nvmlUtilization_t gpuLoad;

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
        return "Could not load GPU usage";

    return std::to_string(gpuLoad.gpu);
}

std::string gpu_memory_controller_load()
{
    nvmlUtilization_t gpuLoad;
    std::stringstream ss;
    ss << "<td style=\"padding-right: 15px\">VRAM controller load</td>";

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU memory load
    float load = static_cast<float>(gpuLoad.memory);
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string gpu_memory_controller_load_as_number()
{
    nvmlUtilization_t gpuLoad;

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
        return "Could not load GPU usage";

    return std::to_string(gpuLoad.memory);
}

std::string gpu_memory()
{
    std::stringstream ss;
    ss << "<td>VRAM</td>";
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    float free_f = static_cast<float>(free);
    float total_f = static_cast<float>(total);

    ss << "<td style=\"color:" << get_load_color(total_f - free_f, total_f) << ";\">" << engineering_notation(free_f, 3)
       << "B free/" << engineering_notation(total_f, 3) << "B</td>";

    return ss.str();
}

} // namespace holovibes