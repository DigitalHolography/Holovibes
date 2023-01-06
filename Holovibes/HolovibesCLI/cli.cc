#include "cli.hh"

#include "chrono.hh"

#include "tools.hh"
#include "icompute.hh"
#include "holovibes_config.hh"
#include "input_frame_file_factory.hh"

#include "global_state_holder.hh"
#include "API.hh"
#include "logger.hh"

namespace holovibes::cli
{
void progress_bar(int current, int total, int length)
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

static void start_worker(const OptionsDescriptor& opts)
{
    api::detail::set_value<IsGuiEnable>(false);
    api::detail::set_value<ImportFilePath>(opts.input_path.value());

    if (opts.compute_settings_path)
        api::load_compute_settings(opts.compute_settings_path.value());

    api::set_end_frame(opts.end_frame.value_or(api::detail::get_value<FileNumberOfFrame>()));
    api::set_start_frame(opts.start_frame.value_or(0));

    api::detail::set_value<LoadFileInGpu>(opts.gpu);
    api::detail::set_value<InputFps>(opts.fps.value_or(DEFAULT_CLI_FPS));
    // api::detail::set_value<LoopFile>(false);

    size_t input_nb_frames = api::get_end_frame() - api::get_start_frame() + 1;
    uint record_nb_frames = opts.n_rec.value_or(input_nb_frames / api::get_time_stride());

    uint nb_frames_skip = 0;
    if (!opts.noskip_acc && api::get_view_xy().is_image_accumulation_enabled())
        nb_frames_skip = api::get_view_xy().output_image_accumulation;

    api::detail::set_value<Record>(
        RecordStruct{opts.output_path.value(),
                     record_nb_frames,
                     nb_frames_skip,
                     true,
                     opts.record_raw ? RecordStruct::RecordType::RAW : RecordStruct::RecordType::HOLOGRAM});

    api::detail::set_value<ImportType>(ImportTypeEnum::File);
}

void start_cli(const OptionsDescriptor& opts)
{
    LOG_FUNC();
    GSH::instance();
    Holovibes::instance();
    Chrono chrono;
    api::check_cuda_graphic_card();

    api::set_input_fps(opts.fps.value_or(DEFAULT_CLI_FPS));

    if (opts.verbose)
        print_verbose(opts);

    LOG_DEBUG("compute_mode = {}", api::detail::get_value<ComputeMode>());

    api::detail::set_value<ComputeMode>(opts.record_raw ? ComputeModeEnum::Raw : ComputeModeEnum::Hologram);

    // FIXME API : maybe this check can go to api
    if (api::get_start_frame() > api::get_end_frame())
    {
        LOG_CRITICAL("-s (start_frame) must be lower or equal than -e (end_frame)");
    }

    start_worker(opts);

    LOG_TRACE("Wait for the record worker to stop ...");
    while (Holovibes::instance().get_frame_record_worker_controller().is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    LOG_TRACE("record worker stopped ...");

    api::detail::set_value<ImportType>(ImportTypeEnum::None);

    LOG_TRACE("Wait for the compute worker to stop ...");
    while (Holovibes::instance().get_compute_worker_controller().is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    LOG_TRACE("compute worker stopped ...");

    LOG_DEBUG("Time: {:.3f}s", chrono.get_milliseconds() / 1000.0f);
}

} // namespace holovibes::cli
