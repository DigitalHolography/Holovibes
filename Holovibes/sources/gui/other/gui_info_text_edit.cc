#include "gui_info_text_edit.hh"

#include <map>
#include <memory>

#include "api.hh"
#include "batch_input_queue.hh"
#include "fast_updates_types.hh"
#include "gpu_stats.hh"
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

// void InfoTextEdit::compute_throughput(size_t output_frame_res, size_t input_frame_size, size_t record_frame_size)
// {
//     input_throughput_ = input_fps_ * input_frame_size;
//     output_throughput_ = output_fps_ * output_frame_res * API.transform.get_time_transformation_size();
//     saving_throughput_ = saving_fps_ * record_frame_size;
// }

// void InfoTextEdit::display_information_slow(size_t elapsed_time)
// {
//     // compute_fps(elapsed_time);
//     std::shared_ptr<Queue> gpu_output_queue = API.compute.get_gpu_output_queue();
//     std::shared_ptr<BatchInputQueue> input_queue = API.compute.get_input_queue();
//     std::shared_ptr<Queue> frame_record_queue = Holovibes::instance().get_record_queue().load();

//     unsigned int output_frame_res = 0;
//     unsigned int input_frame_size = 0;
//     unsigned int record_frame_size = 0;

//     if (gpu_output_queue && input_queue)
//     {
//         output_frame_res = static_cast<unsigned int>(gpu_output_queue->get_fd().get_frame_res());
//         input_frame_size = static_cast<unsigned int>(input_queue->get_fd().get_frame_size());
//     }

//     if (frame_record_queue)
//         record_frame_size = static_cast<unsigned int>(frame_record_queue->get_fd().get_frame_size());

//     compute_throughput(output_frame_res, input_frame_size, record_frame_size);
// }

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

    if (information_.input_fps)
        to_display << "<tr><td>Input FPS</td><td>" << *information_.input_fps << "</td></tr>";

    if (information_.output_fps)
    {
        to_display << "<tr><td>Output FPS</td>";
        if (*information_.output_fps == 0)
            to_display << "<td style=\"color: red;\">" << *information_.output_fps << "</td></tr>";
        else
            to_display << "<td>" << *information_.output_fps << "</td></tr>";
    }

    if (information_.saving_fps)
        to_display << "<tr><td>Saving FPS</td><td>" << *information_.saving_fps << "</td></tr>";

    if (information_.output_fps)
    {
        to_display << "<tr><td>Input Throughput</td><td>" << format_throughput(information_.input_throughput, "B/s")
                   << "</td></tr>";
        to_display << "<tr><td>Output Throughput</td><td>"
                   << format_throughput(information_.output_throughput, "Voxels/s") << "</td></tr>";
    }

    if (information_.saving_fps)
    {
        to_display << "<tr><td>Saving Throughput</td><td>  " << format_throughput(information_.saving_throughput, "B/s")
                   << "</td></tr>";
    }

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