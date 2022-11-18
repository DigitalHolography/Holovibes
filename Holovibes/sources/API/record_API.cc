#include "API.hh"

namespace holovibes::api
{

// FIXME : record seems to be weirdly coded
bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path)
{
    // Preconditions to start record

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || api::get_frame_record_mode().record_mode == RecordMode::CHART) &&
        nb_frames_to_record == std::nullopt)
    {
        LOG_ERROR(main, "Number of frames must be activated");
        return false;
    }

    if (batch_enabled && batch_input_path.empty())
    {
        LOG_ERROR(main, "No batch input file");
        return false;
    }

    return true;
}

void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback)
{
    if (batch_enabled)
    {
        Holovibes::instance().start_batch_gpib(batch_input_path,
                                               output_path,
                                               nb_frames_to_record.value(),
                                               api::get_frame_record_mode().record_mode,
                                               callback);
    }
    else
    {
        if (api::get_frame_record_mode().record_mode == RecordMode::CHART)
        {
            Holovibes::instance().start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else
        {
            Holovibes::instance().start_frame_record(output_path,
                                                     nb_frames_to_record,
                                                     api::get_frame_record_mode().record_mode,
                                                     0,
                                                     callback);
        }
    }
}

void stop_record()
{
    LOG_FUNC(main);

    Holovibes::instance().stop_batch_gpib();

    if (api::get_frame_record_mode().record_mode == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();

    else if (api::get_frame_record_mode().record_mode == RecordMode::HOLOGRAM ||
             api::get_frame_record_mode().record_mode == RecordMode::RAW ||
             api::get_frame_record_mode().record_mode == RecordMode::CUTS_XZ ||
             api::get_frame_record_mode().record_mode == RecordMode::CUTS_YZ)
        Holovibes::instance().stop_frame_record();
}

const std::string browse_record_output_file(std::string& std_filepath)
{
    // FIXME API : This has to be in GUI

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    UserInterfaceDescriptor::instance().record_output_directory_ = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    UserInterfaceDescriptor::instance().default_output_filename_ = path.stem().string();

    return file_ext;
}

} // namespace holovibes::api
