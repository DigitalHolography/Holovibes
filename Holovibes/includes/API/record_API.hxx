#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path)
{
    // Preconditions to start record

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART) &&
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
                                               UserInterfaceDescriptor::instance().record_mode_,
                                               callback);
    }
    else
    {
        if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        {
            Holovibes::instance().start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else
        {
            Holovibes::instance().start_frame_record(output_path,
                                                     nb_frames_to_record,
                                                     UserInterfaceDescriptor::instance().record_mode_,
                                                     0,
                                                     callback);
        }
    }
}

void stop_record()
{
    LOG_FUNC(main);

    Holovibes::instance().stop_batch_gpib();

    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_XZ ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_YZ)
        Holovibes::instance().stop_frame_record();
}

void set_record_mode(const std::string& text)
{
    LOG_FUNC(main, text);

    // TODO: Dictionnary
    if (text == "Chart")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::RAW;
    else if (text == "3D Cuts XZ")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CUTS_XZ;
    else if (text == "3D Cuts YZ")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CUTS_YZ;
    else
        throw std::exception("Record mode not handled");
}

} // namespace holovibes::api
