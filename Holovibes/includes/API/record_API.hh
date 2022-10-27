#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline void record_finished() { UserInterfaceDescriptor::instance().is_recording_ = false; }

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path);

void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback);

void stop_record();

const std::string browse_record_output_file(std::string& std_filepath);

} // namespace holovibes::api
