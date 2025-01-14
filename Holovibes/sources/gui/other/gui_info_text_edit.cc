#include "gui_info_text_edit.hh"

#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nvml.h>
#include <sstream>
#include <string>

#include "api.hh"
#include "batch_input_queue.hh"
#include "fast_updates_types.hh"
#include "queue.hh"

namespace holovibes::gui
{
static const std::unordered_map<QueueType, std::string> queue_type_to_string_ = {
    {QueueType::INPUT_QUEUE, "Input Queue"},
    {QueueType::OUTPUT_QUEUE, "Output Queue"},
    {QueueType::RECORD_QUEUE, "Record Queue"},
};

static std::string format_throughput(size_t throughput, const std::string& unit)
{
    float throughput_ = throughput / (throughput > 1e9f ? 1e9f : 1e6f);
    std::string unit_ = (throughput > 1e9f ? " G" : " M") + unit;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << throughput_ << unit_;

    return ss.str();
}

static std::string get_load_color(float load,
                                  float max_load,
                                  float orange_ratio = ORANGE_COLORATION_RATIO,
                                  float red_ratio = RED_COLORATION_RATIO)
{
    const float ratio = (load / max_load);
    if (ratio < orange_ratio)
        return "white";
    if (ratio < red_ratio)
        return "orange";
    return "red";
}

static std::string get_percentage_color(float percentage) { return get_load_color(percentage, 100); }

std::string InfoTextEdit::gpu_load()
{
    std::stringstream ss;
    ss << "<td>GPU load</td>";

    if (!information_.gpu_info)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU load
    float load = static_cast<float>(information_.gpu_info->gpu);
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string InfoTextEdit::gpu_memory_controller_load()
{
    std::stringstream ss;
    ss << "<td style=\"padding-right: 15px\">VRAM controller load</td>";

    if (!information_.gpu_info)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU memory load
    float load = static_cast<float>(information_.gpu_info->memory);
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string InfoTextEdit::gpu_memory()
{
    std::stringstream ss;
    ss << "<td>VRAM</td>";

    if (!information_.gpu_info)
    {
        ss << "<td>Could not load VRAM info</td>";
        return ss.str();
    }

    float free_f = static_cast<float>(information_.gpu_info->controller_memory);
    float total_f = static_cast<float>(information_.gpu_info->controller_total);

    ss << "<td style=\"color:" << get_load_color(total_f - free_f, total_f) << ";\">" << engineering_notation(free_f, 3)
       << "B free/" << engineering_notation(total_f, 3) << "B</td>";

    return ss.str();
}

void InfoTextEdit::display_information()
{
    information_ = API.information.get_information();

    std::string str;
    str.reserve(512);
    std::stringstream to_display(str);

    to_display << "<table>";

    if (information_.img_source)
    {
        to_display << "<tr><td>Image Source</td><td>" << *information_.img_source.get() << "</td></tr>";
        if (information_.temperature && *information_.temperature != 0)
            to_display << "<tr><td>Camera Temperature</td><td>" << *information_.temperature << "°C</td></tr>";
    }
    if (information_.input_format)
        to_display << "<tr><td>Input Format</td><td>" << *information_.input_format.get() << "</td></tr>";
    if (information_.output_format)
        to_display << "<tr><td>Output Format</td><td>" << *information_.output_format.get() << "</td></tr>";

    if (!API.compute.get_is_computation_stopped())
    {
        for (auto const& [key, info] : information_.queues)
        {
            if (key == QueueType::UNDEFINED)
                continue;
            float currentLoad = static_cast<float>(info.current_size);
            float maxLoad = static_cast<float>(info.max_size);

            to_display << "<tr style=\"color:";
            if (key == QueueType::OUTPUT_QUEUE)
                to_display << "white";
            else if (key == QueueType::INPUT_QUEUE)
                to_display << get_load_color(currentLoad,
                                             maxLoad,
                                             INPUT_Q_ORANGE_COLORATION_RATIO,
                                             INPUT_Q_RED_COLORATION_RATIO);
            else
                to_display << get_load_color(currentLoad, maxLoad);

            to_display << ";\">";

            to_display << "<td>" << (info.device == Device::GPU ? "GPU " : "CPU ") << queue_type_to_string_.at(key)
                       << "</td>";
            to_display << "<td>" << currentLoad << "/" << maxLoad << "</td></tr>";
        }
    }

    if (information_.input)
        to_display << "<tr><td>Input FPS</td><td>" << information_.input->fps << "</td></tr>";

    if (information_.output)
    {
        to_display << "<tr><td>Output FPS</td>";
        if (information_.output->fps == 0)
            to_display << "<td style=\"color: red;\">" << information_.output->fps << "</td></tr>";
        else
            to_display << "<td>" << information_.output->fps << "</td></tr>";
    }

    if (information_.saving)
        to_display << "<tr><td>Saving FPS</td><td>" << information_.saving->fps << "</td></tr>";

    if (information_.input)
        to_display << "<tr><td>Input Throughput</td><td>" << format_throughput(information_.input->throughput, "B/s")
                   << "</td></tr>";
    if (information_.output)
        to_display << "<tr><td>Output Throughput</td><td>"
                   << format_throughput(information_.output->throughput, "Voxels/s") << "</td></tr>";

    if (information_.saving)
        to_display << "<tr><td>Saving Throughput</td><td>  "
                   << format_throughput(information_.saving->throughput, "B/s") << "</td></tr>";

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    to_display << "<tr>" << gpu_memory() << "</tr>";
    /* There is a memory leak on both gpu_load() and gpu_memory_controller_load(), probably linked to nvmlInit */
    to_display << "<tr>" << gpu_load() << "</tr>";
    to_display << "<tr>" << gpu_memory_controller_load() << "</tr>";

    to_display << "</table>";

    this->setText(to_display.str().c_str());
}

} // namespace holovibes::gui