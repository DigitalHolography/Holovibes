#include "cli.hh"

#include <iostream>
#include "chrono.hh"

#include "tools.hh"
#include "icompute.hh"
#include "holovibes_config.hh"
#include "input_frame_file_factory.hh"
#include "enum_record_mode.hh"
#include "enum_window_kind.hh"
#include "fast_updates_holder.hh"
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
    int n = static_cast<int>(length * ratio);

    text += '[';
    if (n == length)
        text.append(n, '=');
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
    LOG_INFO("===== Config =====");
    LOG_INFO("Input file: {}", opts.input_path.value());
    LOG_INFO("Output file: {}", opts.output_path.value());
    LOG_INFO("FPS: {}", opts.fps.value_or(DEFAULT_CLI_FPS));

    LOG_INFO("Number of frames to record: ");
    if (opts.n_rec)
        LOG_INFO("{}", opts.n_rec.value());
    else
        LOG_INFO("full file");

    LOG_INFO("Raw recording: {}", opts.record_raw);
    LOG_INFO("Skip accumulation frames: {}", !opts.noskip_acc);
    LOG_INFO("Number of frames to skip between each frame: {}", opts.frame_skip ? opts.frame_skip.value() : 0);
    LOG_INFO("Number of Mp4 fps: {}", opts.mp4_fps ? opts.mp4_fps.value() : 24);
    LOG_INFO("Moments record: {}", opts.moments_record);
    LOG_INFO("==== End Config ====");
}

int get_first_and_last_frame(const holovibes::OptionsDescriptor& opts, const uint& nb_frames)
{
    auto& api = API;
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
        return 31;
    }

    api.input.set_input_file_start_index(start_frame - 1);

    uint end_frame = opts.end_frame.value_or(nb_frames);
    if (!is_between(end_frame, (uint)1, nb_frames))
    {
        err_message("end_frame", end_frame, "-e");
        return 31;
    }
    api.input.set_input_file_end_index(end_frame);

    if (start_frame > end_frame)
    {
        holovibes::Logger::logger()->error("-s (start_frame) must be lower or equal than -e (end_frame).");
        return 32;
    }

    return 0;
}

static int set_parameters(const holovibes::OptionsDescriptor& opts)
{
    std::string input_path = opts.input_path.value();
    auto& api = API;

    std::optional<holovibes::io_files::InputFrameFile*> input_frame_file_opt = api.input.import_file(input_path, "");
    if (!input_frame_file_opt)
    {
        LOG_ERROR("Failed to open input file");
        return 33;
    }

    holovibes::io_files::InputFrameFile* input_frame_file = input_frame_file_opt.value();

    // To load parameters, we now load before the footer and then the config file so that the config file overwrite the
    // footer
    if (!opts.compute_settings_path && !input_frame_file->get_has_footer())
    {
        LOG_ERROR("No compute settings file provided and no footer found in input file");
        return 35;
    }

    if (opts.compute_settings_path)
    {
        try
        {
            api.settings.load_compute_settings(opts.compute_settings_path.value());
        }
        catch (std::exception& e)
        {
            LOG_INFO(e.what());
            LOG_INFO("Error while loading compute settings, abort");
            return 34;
        }
    }

    if (opts.record_raw && opts.moments_record)
    {
        LOG_ERROR("Cannot record raw and moments at the same time");
        return 36;
    }

    if (opts.record_raw)
    {
        api.compute.set_compute_mode(holovibes::Computation::Raw);
        api.record.set_record_mode(holovibes::RecordMode::RAW);
    }
    else if (opts.moments_record)
        api.record.set_record_mode(holovibes::RecordMode::MOMENTS);
    else
    {
        api.compute.set_compute_mode(holovibes::Computation::Hologram);
        api.record.set_record_mode(holovibes::RecordMode::HOLOGRAM);
    }

    if (int ret = get_first_and_last_frame(opts, static_cast<uint>(input_frame_file->get_total_nb_frames())))
        return ret; // error 31, 32

    delete input_frame_file;

    return 0;
}

static void main_loop()
{
    while (API.record.is_recording())
    {
        if (API.information.get_information().record_info)
        {
            holovibes::RecordProgressInfo progress = API.information.get_information().record_info.value();
            progress_bar(static_cast<int>(progress.saved_frames), static_cast<int>(progress.total_frames), 40);
        }
        else
            LOG_WARN("Unable to retrieve recording information, the progress bar will not be updated");

        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

static int start_cli_workers(const holovibes::OptionsDescriptor& opts)
{
    LOG_INFO("Starting CLI workers");
    auto& api = API;
    // Force some values

    // Value used in more than 1 thread
    size_t input_nb_frames = api.input.get_input_file_end_index() - api.input.get_input_file_start_index();
    uint record_nb_frames;
    if (opts.record_raw)
        record_nb_frames = opts.n_rec.value_or(input_nb_frames);
    else
        record_nb_frames = opts.n_rec.value_or(input_nb_frames / api.transform.get_time_stride());

    if (record_nb_frames <= 0)
    {
        LOG_ERROR("Asking to record {} frames, abort", std::to_string(record_nb_frames));
        return 37;
    }

    // Thread 1
    uint nb_frames_skip = 0;
    // Skip img acc frames to avoid early black frames
    if (!opts.noskip_acc && api.window_pp.get_accumulation_level(holovibes::WindowKind::XYview) > 1 && !opts.record_raw)
        nb_frames_skip = api.window_pp.get_accumulation_level(holovibes::WindowKind::XYview);

    if (opts.fps)
        api.input.set_input_fps(opts.fps.value());

    api.record.set_record_file_path(opts.output_path.value());
    api.record.set_record_frame_count(record_nb_frames);
    api.record.set_record_frame_offset(nb_frames_skip);

    if (opts.frame_skip)
        api.record.set_nb_frame_skip(opts.frame_skip.value());

    if (opts.mp4_fps)
        api.record.set_mp4_fps(opts.mp4_fps.value());

    // Change the fps according to the Mp4Fps value when having to convert in Mp4 format
    if (opts.output_path.value().ends_with(".mp4"))
    {
        // Computing the fps before catching the images so that we can set the frame skip according to the fps
        // wanted
        double input_fps = static_cast<double>(api.input.get_input_fps());
        double time_stride = static_cast<double>(api.transform.get_time_stride());
        double frame_skip = static_cast<double>(api.record.get_nb_frame_skip());
        assert(time_stride != 0);
        double output_fps = input_fps / time_stride;

        if (frame_skip > 0)
            output_fps = output_fps / (frame_skip + 1);

        api.record.set_nb_frame_skip(static_cast<uint>(output_fps * (frame_skip + 1)) / api.record.get_mp4_fps());
    }

    api.record.start_record([]() {});
    api.compute.start();

    return 0;
}

int start_cli(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts)
{
    LOG_INFO("Starting CLI");

    holovibes.is_cli = true;
    if (int ret = set_parameters(opts))
        return ret;

    LOG_INFO("Parameters set");

    if (opts.verbose)
        print_verbose(opts);

    Chrono chrono;
    if (int ret = start_cli_workers(opts))
        return ret;

    LOG_INFO("CLI workers started, main looping");

    main_loop();

    LOG_INFO("Main loop ended");
    LOG_DEBUG("Time: {:.3f}s", chrono.get_milliseconds() / 1000.0f);

    API.compute.stop();

    return 0;
}
} // namespace cli
