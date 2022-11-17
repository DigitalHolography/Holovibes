#include "cli.hh"

#include "chrono.hh"

#include "tools.hh"
#include "icompute.hh"
#include "holovibes_config.hh"
#include "input_frame_file_factory.hh"
#include "enum_record_mode.hh"
#include "global_state_holder.hh"
#include "user_interface_descriptor.hh"
#include "API.hh"
#include "logger.hh"

namespace cli
{
static void progress_bar(int current, int total, int length)
{
    if (total == 0)
        return;

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
    LOG_INFO(main, "Config:");

    LOG_INFO(main, "Input file: {}", opts.input_path.value());
    LOG_INFO(main, "Output file: {}", opts.output_path.value());
    LOG_INFO(main, "FPS: {}", opts.fps.value_or(DEFAULT_CLI_FPS));
    LOG_INFO(main, "Number of frames to record: ");
    if (opts.n_rec)
    {
        LOG_INFO(main, "{}", opts.n_rec.value());
    }
    else
    {
        LOG_INFO(main, "full file");
    }
    LOG_INFO(main, "Raw recording: {}", opts.record_raw);
    LOG_INFO(main, "Skip accumulation frames: {}", !opts.noskip_acc);
    LOG_INFO(main, "Load in GPU: {}", opts.gpu);
}

int get_first_and_last_frame(const holovibes::OptionsDescriptor& opts, const uint& nb_frames)
{
    auto err_message = [&](const std::string& name, const uint& value, const std::string& option)
    {
        spdlog::get("Setup")->error(
            "{} ({}) value: {} is not valid. The valid condition is: 1 <= {} <= nb_frame. For this file nb_frame = ",
            option,
            name,
            value,
            name,
            nb_frames);
    };

    uint start_frame = opts.start_frame.value_or(1);
    if (!is_between(start_frame, (uint)1, nb_frames))
    {
        err_message("start_frame", start_frame, "-s");
        return 2;
    }
    holovibes::api::set_start_frame(start_frame);

    uint end_frame = opts.end_frame.value_or(nb_frames);
    if (!is_between(end_frame, (uint)1, nb_frames))
    {
        err_message("end_frame", end_frame, "-e");
        return 2;
    }
    holovibes::api::set_end_frame(end_frame);

    if (start_frame > end_frame)
    {
        spdlog::get("Setup")->error("-s (start_frame) must be lower or equal than -e (end_frame).");
        return 2;
    }

    return 0;
}

static int set_parameters(const holovibes::OptionsDescriptor& opts)
{
    std::string input_path = opts.input_path.value();

    holovibes::io_files::InputFrameFile* input_frame_file =
        holovibes::io_files::InputFrameFileFactory::open(input_path);

    bool load = false;
    if (input_frame_file->get_has_footer())
    {
        LOG_DEBUG(main, "loading pixel size");
        // Pixel size is set with info section of input file we need to call import_compute_settings in order to load
        // the footer and then import info
        input_frame_file->import_compute_settings();
        input_frame_file->import_info();
        load = true;
    }

    if (opts.compute_settings_path)
    {
        try
        {
            holovibes::api::load_compute_settings(opts.compute_settings_path.value());
        }
        catch (std::exception&)
        {
            LOG_WARN(setup, "Configuration file not found.");
            return 1;
        }
    }
    else if (!load)
        input_frame_file->import_compute_settings();

    const camera::FrameDescriptor& fd = input_frame_file->get_frame_descriptor();

    if (int ret = get_first_and_last_frame(opts, static_cast<uint>(input_frame_file->get_total_nb_frames())))
        return ret;

    holovibes::Holovibes::instance().init_input_queue(fd, holovibes::api::get_input_buffer_size());

    try
    {
        holovibes::Holovibes::instance().init_pipe();
    }
    catch (std::exception& e)
    {
        LOG_ERROR(setup, "{}", e.what());
        return 1;
    }

    if (holovibes::GSH::instance().get_value<holovibes::Convolution>().enabled)
        holovibes::api::enable_convolution(holovibes::api::detail::get_value<holovibes::Convolution>().type);

    // WHY Trigger ?
    holovibes::api::detail::change_value<holovibes::TimeStride>().trigger();
    holovibes::api::detail::change_value<holovibes::TimeTransformationSize>().trigger();

    delete input_frame_file;

    return 0;
}

static void main_loop()
{
    // Recording progress (used by the progress bar)
    holovibes::FastUpdatesHolder<holovibes::ProgressType>::Value progress = nullptr;

    // Request auto contrast once if auto refresh is enabled
    bool requested_autocontrast = holovibes::GSH::instance().get_value<holovibes::ViewXY>().contrast.auto_refresh;

    while (holovibes::GSH::instance().get_value<holovibes::FrameRecordMode>().enabled)
    {
        if (holovibes::GSH::fast_updates_map<holovibes::ProgressType>.contains(holovibes::ProgressType::FRAME_RECORD))
        {
            if (!progress)
                progress = holovibes::GSH::fast_updates_map<holovibes::ProgressType>.get_entry(
                    holovibes::ProgressType::FRAME_RECORD);
            else
            {
                progress_bar(progress->first, progress->second, 40);

                // FIXME : WTF why it is in the cli.cc
                // Very dirty hack
                // Request auto contrast once we have accumualated enough images
                // Otherwise the autocontrast is computed at the beginning and we
                // end up with black images ...
                if (progress->first >= holovibes::api::get_view_xy().img_accu_level && requested_autocontrast)
                {
                    holovibes::api::request_exec_contrast_current_window();
                    requested_autocontrast = false;
                }
            }
        }

        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

static int start_cli_workers(const holovibes::OptionsDescriptor& opts)
{
    // FIXME check for better getters
    // Force some values
    holovibes::Holovibes::instance().is_cli = true;
    holovibes::api::change_frame_record_mode()->record_mode =
        opts.record_raw ? holovibes::RecordMode::RAW : holovibes::RecordMode::HOLOGRAM;
    holovibes::GSH::instance().set_value<holovibes::ComputeMode>(opts.record_raw ? holovibes::Computation::Raw
                                                                                 : holovibes::Computation::Hologram);

    // Value used in more than 1 thread
    size_t input_nb_frames = holovibes::api::get_end_frame() - holovibes::api::get_start_frame() + 1;
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / holovibes::api::get_time_stride());
    if (record_nb_frames == 0)
    {
        LOG_ERROR(setup, "Asking to record 0 frames, abort");
        return 2;
    }

    // Thread 1
    uint nb_frames_skip = 0;
    // Skip img acc frames to avoid early black frames
    if (!opts.noskip_acc && holovibes::api::get_view_xy().is_image_accumulation_enabled())
        nb_frames_skip = holovibes::api::get_view_xy().img_accu_level;

    holovibes::Holovibes::instance().start_frame_record(opts.output_path.value(),
                                                        record_nb_frames,
                                                        opts.record_raw ? holovibes::RecordMode::RAW
                                                                        : holovibes::RecordMode::HOLOGRAM,
                                                        nb_frames_skip);

    // FIXME : this is a dangerous hack need to be changed
    // The following while ensure the record has been requested by the thread previously launched.
    while (!holovibes::api::get_compute_pipe().get_export_cache().has_change_requested())
        continue;

    // The pipe has to be refresh before lauching the next thread to prevent concurrency problems.
    // It has to be refresh in the main thread because the read of file is launched just after.
    holovibes::api::get_compute_pipe().sync_and_refresh();

    // Thread 2
    holovibes::Holovibes::instance().start_compute_worker();

    // Thread 3
    holovibes::Holovibes::instance().start_file_frame_read(opts.input_path.value(),
                                                           true,
                                                           opts.fps.value_or(DEFAULT_CLI_FPS),
                                                           holovibes::api::get_start_frame() - 1,
                                                           static_cast<uint>(input_nb_frames),
                                                           opts.gpu);

    return 0;
}

int start_cli(const holovibes::OptionsDescriptor& opts)
{
    if (int ret = set_parameters(opts))
        return ret;

    if (opts.verbose)
        print_verbose(opts);

    Chrono chrono;

    if (int ret = start_cli_workers(opts))
        return ret;

    main_loop();

    LOG_DEBUG(main, "Time: {:.3f}s", chrono.get_milliseconds() / 1000.0f);

    holovibes::Holovibes::instance().stop_all_worker_controller();

    return 0;
}
} // namespace cli
