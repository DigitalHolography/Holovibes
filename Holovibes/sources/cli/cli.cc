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

namespace holovibes::cli
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

static void print_verbose(const OptionsDescriptor& opts)
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

// FIXME
static void main_loop(const OptionsDescriptor& opts)
{
    // Recording progress (used by the progress bar)
    FastUpdatesHolder<ProgressType>::Value progress = nullptr;

    auto fast_update_progress_entry = GSH::fast_updates_map<ProgressType>.get_entry(ProgressType::FRAME_RECORD);
    std::atomic<uint>& nb_frames_recorded = fast_update_progress_entry->first;

    size_t input_nb_frames = api::get_end_frame() - api::get_start_frame() + 1;
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / api::get_time_stride());

    while (api::detail::get_value<FrameRecordMode>().enabled && nb_frames_recorded < record_nb_frames)
    {
        if (GSH::fast_updates_map<ProgressType>.contains(ProgressType::FRAME_RECORD))
        {
            if (!progress)
                progress = GSH::fast_updates_map<ProgressType>.get_entry(ProgressType::FRAME_RECORD);
            else
                progress_bar(progress->first, progress->second, 40);
        }

        // Don't make the current thread loop too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Show 100% completion to avoid rounding errors
    progress_bar(1, 1, 40);
}

// FIXME
static void start_worker(const OptionsDescriptor& opts)
{
    api::import_file(opts.input_path.value());

    Holovibes::instance().init_input_queue(api::detail::get_value<ImportFrameDescriptor>(),
                                           holovibes::api::get_input_buffer_size());

    Holovibes::instance().init_pipe();

    Holovibes::instance().start_compute_worker();

    Holovibes::instance().start_file_frame_read(opts.input_path.value(),
                                                true,
                                                opts.fps.value_or(DEFAULT_CLI_FPS),
                                                api::get_start_frame() - 1,
                                                api::detail::get_value<FileNumberOfFrame>(),
                                                opts.gpu);

    api::set_start_frame(opts.start_frame.value_or(0));
    api::set_end_frame(opts.end_frame.value_or(api::detail::get_value<FileNumberOfFrame>()));

    size_t input_nb_frames = api::get_end_frame() - api::get_start_frame() + 1;
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / api::get_time_stride());

    uint nb_frames_skip = 0;
    if (!opts.noskip_acc && api::get_view_xy().is_image_accumulation_enabled())
        nb_frames_skip = api::get_view_xy().img_accu_level;
    Holovibes::instance().start_frame_record(opts.output_path.value(),
                                             record_nb_frames,
                                             opts.record_raw ? RecordMode::RAW : RecordMode::HOLOGRAM,
                                             nb_frames_skip);
}

void start_cli(const OptionsDescriptor& opts)
{
    GSH::instance();
    Chrono chrono;
    Holovibes::instance().is_cli = true;
    api::check_cuda_graphic_card();

    start_worker(opts);

    if (opts.verbose)
        print_verbose(opts);

    if (opts.compute_settings_path)
        api::load_compute_settings(opts.compute_settings_path.value());

    // FIXME API : maybe this check can go to api
    if (api::get_start_frame() > api::get_end_frame())
    {
        LOG_CRITICAL(main,
                     "-s (start_frame) must be lower or equal than -e (end_frame) ; btw {} > {} ?",
                     api::get_start_frame(),
                     api::get_end_frame());
    }

    api::set_input_fps(opts.fps.value_or(DEFAULT_CLI_FPS));

    api::detail::set_value<ComputeMode>(opts.record_raw ? Computation::Raw : Computation::Hologram);

    api::change_frame_record_mode()->record_mode = opts.record_raw ? RecordMode::RAW : RecordMode::HOLOGRAM;

    main_loop(opts);

    LOG_DEBUG(main, "Time: {:.3f}s", chrono.get_milliseconds() / 1000.0f);
}
} // namespace holovibes::cli
