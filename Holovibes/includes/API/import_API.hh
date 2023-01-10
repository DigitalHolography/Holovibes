#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline ImportTypeEnum get_import_type() { return api::detail::get_value<ImportType>(); }
inline void set_import_type(ImportTypeEnum value) { api::detail::set_value<ImportType>(value); }

inline const FrameDescriptor& get_import_frame_descriptor() { return api::detail::get_value<ImportFrameDescriptor>(); }
inline TriggerChangeValue<FrameDescriptor> change_import_frame_descriptor()
{
    return api::detail::change_value<ImportFrameDescriptor>();
}

inline const std::string& get_import_file_path() { return api::detail::get_value<ImportFilePath>(); }
inline void set_import_file_path(const std::string& filename)
{
    return api::detail::set_value<ImportFilePath>(filename);
}

inline bool get_load_in_gpu() { return api::detail::get_value<LoadFileInGpu>(); }
inline void set_load_in_gpu(bool value) { api::detail::set_value<LoadFileInGpu>(value); }

inline uint get_start_frame() { return api::detail::get_value<StartFrame>(); }
inline void set_start_frame(uint value) { api::detail::set_value<StartFrame>(value); }

inline uint get_end_frame() { return api::detail::get_value<EndFrame>(); }
inline void set_end_frame(uint value) { api::detail::set_value<EndFrame>(value); }

inline uint get_nb_frame_to_read() { return api::get_end_frame() - api::get_start_frame() + 1; }

inline uint get_file_number_of_frames() { return api::detail::get_value<FileNumberOfFrame>(); }
inline void set_file_number_of_frames(uint value) { api::detail::set_value<FileNumberOfFrame>(value); }

inline bool get_loop_file() { return api::detail::get_value<LoopFile>(); }
inline void set_loop_file(bool value) { api::detail::set_value<LoopFile>(value); }

inline uint get_input_fps() { return api::detail::get_value<InputFps>(); }
inline void set_input_fps(uint value) { api::detail::set_value<InputFps>(value); }

inline CameraKind get_current_camera_kind() { return api::detail::get_value<CurrentCameraKind>(); }
inline void set_current_camera_kind(CameraKind value) { api::detail::set_value<CurrentCameraKind>(value); }

} // namespace holovibes::api
