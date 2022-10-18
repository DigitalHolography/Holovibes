#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void record_finished() { UserInterfaceDescriptor::instance().is_recording_ = false; }

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path);

void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback);

void set_record_mode(const std::string& text);

void stop_record();

} // namespace holovibes::api

#include "record_API.hxx"
