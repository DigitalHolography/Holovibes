#include "cli.hh"

#include <iostream>
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
#include "spdlog/spdlog.h"

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
    LOG_INFO("Config:");

    LOG_INFO("Input file: {}", opts.input_path.value());
    LOG_INFO("Output file: {}", opts.output_path.value());
    LOG_INFO("FPS: {}", opts.fps.value_or(DEFAULT_CLI_FPS));
    LOG_INFO("Number of frames to record: ");
    if (opts.n_rec)
    {
        LOG_INFO("{}", opts.n_rec.value());
    }
    else
    {
        LOG_INFO("full file");
    }
    LOG_INFO("Raw recording: {}", opts.record_raw);
    LOG_INFO("Skip accumulation frames: {}", !opts.noskip_acc);
    LOG_INFO("Load in GPU: {}", opts.gpu);
}

int get_first_and_last_frame(const holovibes::OptionsDescriptor& opts, const uint& nb_frames)
{
    auto err_message = [&](const std::string& name, const uint& value, const std::string& option)
    {
        holovibes::Logger::logger()->error(
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
    holovibes::api::set_input_file_start_index(start_frame - 1);

    uint end_frame = opts.end_frame.value_or(nb_frames);
    if (!is_between(end_frame, (uint)1, nb_frames))
    {
        err_message("end_frame", end_frame, "-e");
        return 2;
    }
    holovibes::api::set_input_file_end_index(end_frame);

    if (start_frame > end_frame)
    {
        holovibes::Logger::logger()->error("-s (start_frame) must be lower or equal than -e (end_frame).");
        return 2;
    }

    return 0;
}

static int set_parameters(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    std::string input_path = opts.input_path.value();
    holovibes::api::set_input_file_path(input_path);
    
    holovibes::io_files::InputFrameFile* input_frame_file = holovibes::io_files::InputFrameFileFactory::open(input_path);
    if (!input_frame_file)
    {
        LOG_ERROR("Failed to open input file");
        return 1;
    }

    

    bool load = false;
    if (input_frame_file->get_has_footer())
    {
        LOG_DEBUG("loading pixel size");
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
        catch (std::exception& e)
        {
            LOG_DEBUG(e.what());
            return 2;
        }
    }
    else if (!load)
    {
        LOG_DEBUG("No compute settings file provided and no footer found in input file");
        return 2;
    }

    const camera::FrameDescriptor& fd = input_frame_file->get_frame_descriptor();

    if (int ret = get_first_and_last_frame(opts, static_cast<uint>(input_frame_file->get_total_nb_frames())))
        return ret;

    holovibes.init_input_queue(fd, holovibes::api::get_input_buffer_size());

    try
    {
        holovibes.init_pipe();
    }
    catch (std::exception& e)
    {
        LOG_ERROR("{}", e.what());
        return 1;
    }

    auto pipe = holovibes.get_compute_pipe();
    if (holovibes::api::get_convolution_enabled())
    {
        holovibes::api::load_convolution_matrix(holovibes::UserInterfaceDescriptor::instance().convo_name);
        pipe->request_convolution();
    }

    // TODO : Add filter

    pipe->request_update_batch_size();
    pipe->request_update_time_stride();
    pipe->request_update_time_transformation_size();

    delete input_frame_file;

    return 0;
}

static void main_loop(holovibes::Holovibes& holovibes)
{
    // Recording progress (used by the progress bar)
    holovibes::FastUpdatesHolder<holovibes::ProgressType>::Value progress = nullptr;

    // Request auto contrast once if auto refresh is enabled
    bool requested_autocontrast = !holovibes::api::get_xy_contrast_auto_refresh();

    while (holovibes::api::get_frame_record_enabled())
    {
        if (holovibes::GSH::fast_updates_map<holovibes::ProgressType>.contains(holovibes::ProgressType::FRAME_RECORD))
        {
            if (!progress)
                progress = holovibes::GSH::fast_updates_map<holovibes::ProgressType>.get_entry(
                    holovibes::ProgressType::FRAME_RECORD);
            else
            {
                progress_bar(progress->first, progress->second, 40);

                // Very dirty hack
                // Request auto contrast once we have accumualated enough images
                // Otherwise the autocontrast is computed at the beginning and we
                // end up with black images ...
                if (progress->first >= holovibes::api::get_xy_accumulation_level() && !requested_autocontrast)
                {
                    holovibes.get_compute_pipe()->request_autocontrast(holovibes::api::get_current_window_type());
                    requested_autocontrast = true;
                }
            }
        }

        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

static int start_cli_workers(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    LOG_INFO("Starting CLI workers");
    // Force some values
    holovibes.is_cli = true;
    auto mode = opts.record_raw ? holovibes::RecordMode::RAW : holovibes::RecordMode::HOLOGRAM;
    holovibes.update_setting(holovibes::settings::RecordMode{mode});
    holovibes::api::set_frame_record_enabled(true);
    holovibes::api::set_compute_mode(opts.record_raw ? holovibes::Computation::Raw : holovibes::Computation::Hologram);

    // Value used in more than 1 thread
    size_t input_nb_frames =
        holovibes::api::get_input_file_end_index() - holovibes::api::get_input_file_start_index();
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / holovibes::api::get_time_stride());
    if (record_nb_frames <= 0)
    {
        LOG_ERROR("Asking to record {} frames, abort", std::to_string(record_nb_frames));
        return 2;
    }

    // Thread 1
    uint nb_frames_skip = 0;
    // Skip img acc frames to avoid early black frames
    if (!opts.noskip_acc && holovibes::api::get_xy_img_accu_enabled())
        nb_frames_skip = holovibes::api::get_xy_accumulation_level();

    holovibes.update_setting(holovibes::settings::RecordFilePath{opts.output_path.value()});
    holovibes.update_setting(holovibes::settings::RecordFrameCount{record_nb_frames});
    holovibes.update_setting(holovibes::settings::RecordFrameSkip{nb_frames_skip});

    holovibes::api::set_record_mode(opts.record_raw ? holovibes::RecordMode::RAW : holovibes::RecordMode::HOLOGRAM);
    
    holovibes.start_frame_record();

    // The following while ensure the record has been requested by the thread previously launched.
    while ((!holovibes.get_compute_pipe()->get_frame_record_requested()))
        continue;

    // The pipe has to be refresh before lauching the next thread to prevent concurrency problems.
    // It has to be refresh in the main thread because the read of file is launched just after.
    holovibes.get_compute_pipe()->refresh();

    // Thread 2
    holovibes.start_compute_worker();

    // Thread 3
    holovibes.start_file_frame_read();

    return 0;
}

int start_cli(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    if (int ret = set_parameters(holovibes, opts))
        return ret;

    if (opts.verbose)
        print_verbose(opts);

    Chrono chrono;
    if (int ret = start_cli_workers(holovibes, opts))
        return ret;
    main_loop(holovibes);

    LOG_DEBUG("Time: {:.3f}s", chrono.get_milliseconds() / 1000.0f);
    holovibes.stop_all_worker_controller();
    return 0;
}
} // namespace cli
