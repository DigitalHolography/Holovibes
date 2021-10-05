#include "cli.hh"

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

static void print_verbose(const holovibes::OptionsDescriptor& opts, const holovibes::ComputeDescriptor& cd)
{
    std::cout << "Config:\n";
    boost::property_tree::ptree ptree;
    holovibes::ini::save_ini(ptree, cd);
    boost::property_tree::write_ini(std::cout, ptree);
    std::cout << std::endl;

    std::cout << "Input file: " << opts.input_path.value() << "\n";
    std::cout << "Output file: " << opts.output_path.value() << "\n";
    std::cout << "FPS: " << opts.fps.value_or(60) << "\n";
    std::cout << "Number of frames to record: ";
    if (opts.n_rec)
        std::cout << opts.n_rec.value() << "\n";
    else
        std::cout << "full file\n";
    std::cout << "Raw recording: " << std::boolalpha << opts.record_raw << std::dec << "\n";
    if (opts.convo_path.has_value())
    {
        std::cout << "Convolution matrix: " << opts.convo_path.value() << "\n";
        std::cout << "Divide by convolution matrix: " << std::boolalpha << opts.divide_convo << std::dec << "\n";
    }
    std::cout << "Skip accumulation frames: " << std::boolalpha << !opts.noskip_acc << std::dec << "\n";
    std::cout << "Load in GPU: " << std::boolalpha << opts.gpu << std::dec << "\n";
    std::cout << std::endl;
}

void get_first_and_last_frame(const holovibes::OptionsDescriptor& opts,
                              const uint& nb_frames,
                              holovibes::ComputeDescriptor& cd)
{
    auto err_message = [&](const std::string& name, const uint& value, const std::string& option) {
        std::cerr << option << " (" << name << ") value: " << value
                  << " is not valid. The valid condition is: 1 <= " << name
                  << " <= nb_frame. For this file nb_frame = " << nb_frames << ".";
    };

    uint start_frame = opts.start_frame.value_or(1);
    if (!is_between(start_frame, (uint)1, nb_frames))
    {
        err_message("start_frame", start_frame, "-s");
        exit(1);
    }
    cd.start_frame = start_frame;

    uint end_frame = opts.end_frame.value_or(nb_frames);
    if (!is_between(end_frame, (uint)1, nb_frames))
    {
        err_message("end_frame", end_frame, "-e");
        exit(1);
    }
    cd.end_frame = end_frame;

    if (start_frame > end_frame)
    {
        std::cerr << "-s (start_frame) must be lower or equal than -e (end_frame).";
        exit(1);
    }
}

static void set_parameters(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    auto& cd = holovibes.get_cd();

    std::string input_path = opts.input_path.value();

    holovibes::io_files::InputFrameFile* input_frame_file =
        holovibes::io_files::InputFrameFileFactory::open(input_path);

    input_frame_file->import_compute_settings(holovibes.get_cd());

    const camera::FrameDescriptor& fd = input_frame_file->get_frame_descriptor();

    get_first_and_last_frame(opts, input_frame_file->get_total_nb_frames(), cd);

    const unsigned int fps = opts.fps.value_or(60);

    if (opts.convo_path.has_value())
    {
        auto convo_path = std::filesystem::path(opts.convo_path.value()).filename().string();
        cd.set_convolution(true, convo_path);
        cd.set_divide_by_convo(opts.divide_convo);
        holovibes.get_compute_pipe()->request_convolution();
    }

    holovibes.init_input_queue(fd);

    try
    {
        holovibes.init_pipe();
    }
    catch (std::exception& e)
    {
        LOG_ERROR << e.what();
        return;
    }

    auto pipe = holovibes.get_compute_pipe();
    pipe->request_update_batch_size();
    pipe->request_update_time_transformation_stride();
    pipe->request_update_time_transformation_size();
    pipe->request_refresh();

    delete input_frame_file;
}

static void main_loop(holovibes::Holovibes& holovibes)
{
    auto& cd = holovibes.get_cd();
    const auto& info = holovibes.get_info_container();
    // Recording progress (used by the progress bar)
    auto record_progress = info.get_progress_index(holovibes::InformationContainer::ProgressType::FRAME_RECORD);

    // Request auto contrast once if auto refresh is enabled
    bool requested_autocontrast = !cd.contrast_auto_refresh;
    while (cd.frame_record_enabled)
    {
        if (!record_progress)
            record_progress = info.get_progress_index(holovibes::InformationContainer::ProgressType::FRAME_RECORD);
        else
        {
            const auto& progress = record_progress.value();
            progress_bar(progress.first->load(), progress.second->load(), 40);

            // Very dirty hack
            // Request auto contrast once we have accumualated enough images
            // Otherwise the autocontrast is computed at the beginning and we
            // end up with black images ...
            if (progress.first->load() >= cd.img_acc_slice_xy_level && !requested_autocontrast)
            {
                holovibes.get_compute_pipe()->request_autocontrast(cd.current_window);
                requested_autocontrast = true;
            }
        }
        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

int start_cli(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    auto& cd = holovibes.get_cd();

    if (opts.ini_path)
    {
        try
        {
            holovibes::ini::load_ini(cd, opts.ini_path.value());
        }
        catch (std::exception&)
        {
            LOG_WARN << "Configuration file not found, initialization with default values.";
        }
    }

    set_parameters(holovibes, opts);

    size_t input_nb_frames = cd.end_frame - cd.start_frame + 1;
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / cd.time_transformation_stride);

    // Force hologram mode
    cd.compute_mode = holovibes::Computation::Hologram;

    holovibes.start_information_display(true); // Thread

    Chrono chrono;

    uint nb_frames_skip = 0;

    // Skip img acc frames to avoid early black frames
    if (!opts.noskip_acc && cd.img_acc_slice_xy_enabled)
        nb_frames_skip = cd.img_acc_slice_xy_level;

    cd.frame_record_enabled = true;

    holovibes.start_cli_compute_and_record(opts.output_path.value(),
                                           record_nb_frames,
                                           opts.record_raw,
                                           nb_frames_skip,
                                           [&]() {

                                           });

    holovibes.start_file_frame_read(opts.input_path.value(),
                                    true,
                                    opts.fps.value_or(60),
                                    cd.start_frame - 1,
                                    input_nb_frames,
                                    opts.gpu);

    if (opts.verbose)
    {
        print_verbose(opts, cd);
    }

    main_loop(holovibes);

    printf(" Time: %.3fs\n", chrono.get_milliseconds() / 1000.0f);

    holovibes.stop_all_worker_controller();

    return 0;
}
} // namespace cli
