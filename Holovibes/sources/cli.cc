#include "cli.hh"

// #include <chrono>
#include "chrono.hh"

#include "tools.hh"
#include "icompute.hh"
#include "ini_config.hh"
#include "input_frame_file_factory.hh"

namespace cli
{
static void progress_bar(int current, int total, int length)
{
    std::string text;
    text.reserve(length + 2);

    float ratio = (current * 1.0f) / total;
    int n = length * ratio;

    text += '[';
    if (n == length)
    {
        text.append(n, '=');
    }
    else if (n > 0)
    {
        text.append(n - 1, '=');
        text.append(1, '>');
    }
    text.append(length - n, ' ');
    text += ']';

    std::cout << '\r' << text;
    std::cout.flush();
}

static void print_verbose(const holovibes::OptionsDescriptor& opts)
{
    std::cout << "Config:\n";
    auto ini_data =
        read_file<std::string>(opts.ini_path.value_or(GLOBAL_INI_PATH));
    std::cout << ini_data << "\n\n";

    std::cout << "Input file: " << opts.input_path.value() << "\n";
    std::cout << "Output file: " << opts.output_path.value() << "\n";
    std::cout << "FPS: " << opts.fps.value_or(60) << "\n";
    std::cout << "Number of frames to record: ";
    if (opts.n_rec)
        std::cout << opts.n_rec.value() << "\n";
    else
        std::cout << "full file\n";
    if (opts.record_raw)
        std::cout << "Raw recording enabled\n";
    if (opts.convo_path.has_value())
    {
        std::cout << "Convolution matrix: " << opts.convo_path.value() << "\n";
        std::cout << "Divide by convolution matrix: " << std::boolalpha
                  << opts.divide_convo << std::dec << "\n";
    }
    std::cout << "Skip accumulation frames: " << std::boolalpha
              << !opts.noskip_acc << std::dec << "\n";
    std::cout << std::endl;
}

static holovibes::io_files::InputFrameFile*
open_input_file(holovibes::Holovibes& holovibes,
                const holovibes::OptionsDescriptor& opts)
{
    std::string input_path = opts.input_path.value();

    holovibes::io_files::InputFrameFile* input_frame_file =
        holovibes::io_files::InputFrameFileFactory::open(input_path);

    const camera::FrameDescriptor& fd =
        input_frame_file->get_frame_descriptor();

    const unsigned int fps = opts.fps.value_or(60);
    holovibes.init_input_queue(fd);
    holovibes.start_file_frame_read(input_path,
                                    true,
                                    fps,
                                    0,
                                    input_frame_file->get_total_nb_frames(),
                                    false);

    input_frame_file->import_compute_settings(holovibes.get_cd());

    return input_frame_file;
}

static void set_parameters(holovibes::Holovibes& holovibes,
                           const holovibes::OptionsDescriptor& opts)
{
    auto& cd = holovibes.get_cd();
    if (opts.convo_path.has_value())
    {
        auto convo_path =
            std::filesystem::path(opts.convo_path.value()).filename().string();
        cd.set_convolution(true, convo_path);
        cd.set_divide_by_convo(opts.divide_convo);
        holovibes.get_compute_pipe()->request_convolution();
    }

    holovibes.get_compute_pipe()->request_update_batch_size();
    holovibes.get_compute_pipe()->request_update_time_transformation_stride();
    holovibes.get_compute_pipe()->request_update_time_transformation_size();
    holovibes.get_compute_pipe()->request_refresh();
}

static void start_record(holovibes::Holovibes& holovibes,
                         const holovibes::OptionsDescriptor& opts,
                         size_t input_nb_frames)
{
    auto& cd = holovibes.get_cd();
    uint nb_frames_skip = 0;
    // Skip img acc frames to avoid early black frames
    if (!opts.noskip_acc && cd.img_acc_slice_xy_enabled)
    {
        nb_frames_skip = cd.img_acc_slice_xy_level;
    }
    holovibes.start_frame_record(opts.output_path.value(),
                                 opts.n_rec.value_or(input_nb_frames),
                                 opts.record_raw,
                                 false,
                                 nb_frames_skip);
}

static void main_loop(holovibes::Holovibes& holovibes)
{
    auto& cd = holovibes.get_cd();
    const auto& info = holovibes.get_info_container();
    // Recording progress (used by the progress bar)
    auto record_progress = info.get_progress_index(
        holovibes::InformationContainer::ProgressType::FRAME_RECORD);

    // Request auto contrast once if auto refresh is enabled
    bool requested_autocontrast = !cd.contrast_auto_refresh;
    while (cd.frame_record_enabled)
    {
        if (!record_progress)
            record_progress = info.get_progress_index(
                holovibes::InformationContainer::ProgressType::FRAME_RECORD);
        else
        {
            const auto& progress = record_progress.value();
            progress_bar(progress.first->load(), progress.second->load(), 40);

            // Very dirty hack
            // Request auto contrast once we have accumualated enough images
            // Otherwise the autocontrast is computed at the beginning and we
            // end up with black images ...
            if (progress.first->load() >= cd.img_acc_slice_xy_level &&
                !requested_autocontrast)
            {
                holovibes.get_compute_pipe()->request_autocontrast(
                    cd.current_window);
                requested_autocontrast = true;
            }
        }
        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

int start_cli(holovibes::Holovibes& holovibes,
              const holovibes::OptionsDescriptor& opts)
{
    if (opts.verbose)
    {
        print_verbose(opts);
    }

    std::string ini_path = opts.ini_path.value_or(GLOBAL_INI_PATH);
    holovibes::ini::load_ini(holovibes.get_cd(), ini_path);
    holovibes.start_information_display(true);

    auto& cd = holovibes.get_cd();
    cd.compute_mode = holovibes::Computation::Hologram;
    cd.frame_record_enabled = true;

    auto input_frame_file = open_input_file(holovibes, opts);
    size_t input_nb_frames = input_frame_file->get_total_nb_frames();

    // Force hologram mode :
    // .holo meta data can reset compute mode to raw
    cd.compute_mode = holovibes::Computation::Hologram;
    cd.frame_record_enabled = true;

    Chrono chrono;

    holovibes.start_compute();
    set_parameters(holovibes, opts);
    start_record(holovibes, opts, input_nb_frames);
    main_loop(holovibes);

    chrono.stop();

    printf(" Time: %.3fs\n", chrono.get_milliseconds() / 1000.0f);

    holovibes.stop_all_worker_controller();

    delete input_frame_file;

    return 0;
}
} // namespace cli
