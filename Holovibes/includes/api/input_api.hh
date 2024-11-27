/*! \file
 *
 * \brief Regroup all functions used to interact with file loading and camera managment.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

inline size_t get_input_file_start_index() { return GET_SETTING(InputFileStartIndex); }
inline size_t get_input_file_end_index() { return GET_SETTING(InputFileEndIndex); }

inline std::string get_input_file_path() { return GET_SETTING(InputFilePath); }
inline void set_input_file_path(std::string value) { UPDATE_SETTING(InputFilePath, value); }

inline bool get_load_file_in_gpu() { return GET_SETTING(LoadFileInGPU); }
inline void set_load_file_in_gpu(bool value) { UPDATE_SETTING(LoadFileInGPU, value); }

inline uint get_input_fps() { return static_cast<uint>(GET_SETTING(InputFPS)); }
inline void set_input_fps(uint value) { UPDATE_SETTING(InputFPS, value); }

inline camera::FrameDescriptor get_input_fd() { return GET_SETTING(ImportedFileFd); }
inline void set_input_fd(camera::FrameDescriptor value) { UPDATE_SETTING(ImportedFileFd, value); }

inline ImportType get_import_type() { return GET_SETTING(ImportType); }
inline void set_import_type(ImportType value) { UPDATE_SETTING(ImportType, value); }

inline CameraKind get_camera_kind() { return GET_SETTING(CameraKind); }
inline void set_camera_kind(CameraKind value) { UPDATE_SETTING(CameraKind, value); }

inline uint get_camera_fps() { return GET_SETTING(CameraFps); }
inline void set_camera_fps(uint value) { UPDATE_SETTING(CameraFps, value); }

/*! \brief Sets the file start index
 *
 * \param[in] value the new value
 */
void set_input_file_start_index(size_t value);

/*! \brief Sets the file end index
 *
 * \param[in] value the new value
 */
void set_input_file_end_index(size_t value);

#pragma region File Import

/*! \brief Launchs the reading of the loaded file
 *
 * \return true on success
 * \return false on failure
 */
bool import_start();

/*! \brief Stops the display */
void import_stop();

/*! \brief Gets an Input file from a given filename
 *
 * \param[in] filename the given filename to open
 *
 * \return std::optional<io_files::InputFrameFile*> the file on success, nullopt on error
 */
std::optional<io_files::InputFrameFile*> import_file(const std::string& filename);

#pragma endregion

#pragma region Cameras

/*! \brief Switchs operating camera to none */
void camera_none();

/*! \brief Configures the camera */
void configure_camera();

/*! \brief Changes the current camera used
 *
 * \param[in] c the camera kind selection.
 * \return true on success
 */
bool change_camera(CameraKind c);

#pragma endregion

} // namespace holovibes::api