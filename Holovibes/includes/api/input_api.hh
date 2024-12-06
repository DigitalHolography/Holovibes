/*! \file input_api.hh
 *
 * \brief Regroup all functions used to interact with file loading and camera managment.
 */
#pragma once

#include "API.hh"

namespace holovibes::api
{

#pragma region Input

/*! \brief Return the current frame descriptor used. A file must be loaded or a camera must be running in order to have
 * a valid frame descriptor.
 *
 * \return const camera::FrameDescriptor& the current frame descriptor
 */
inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_input_queue()->get_fd(); };

/*! \brief Return from where frames came from (camera, file, etc.) or none if no import has been setup.
 *
 * \return ImportType the import type
 */
inline ImportType get_import_type() { return GET_SETTING(ImportType); }

/*! \brief Set from where frames came from (camera, file, etc.) or none if no import has been setup.
 *
 * \param[in] value the new import type
 */
inline void set_import_type(ImportType value) { UPDATE_SETTING(ImportType, value); }

/*! \brief Gets the data type of the recorded file. (whether it was recorded as raw, moment, etc.).
 *
 * \return RecordedDataType the data type
 */
inline RecordedDataType get_data_type() { return GET_SETTING(DataType); }

/*! \brief Sets the data type of the recorded file. (whether it's recorded as raw, moment, etc.).
 *
 * \param[in] data_type the new data type
 */
inline void set_data_type(const RecordedDataType data_type) { UPDATE_SETTING(DataType, data_type); }

#pragma endregion

#pragma region Buffer Size

/*! \brief Return the capacity (number of images) of the input queue.
 *
 * \return uint the capacity of the input queue
 */
inline uint get_input_buffer_size() { return static_cast<uint>(GET_SETTING(InputBufferSize)); }

/*! \brief Set the capacity (number of images) of the input queue. Generally setting it to four times the batch size is
 * sufficient.
 *
 * \param[in] value the new capacity of the input queue
 */
inline void set_input_buffer_size(uint value) { UPDATE_SETTING(InputBufferSize, value); }

/*! \brief Return the number of frames read at once from the file.
 *
 * \return uint the number of frames read at once from the file
 */
inline uint get_file_buffer_size() { return static_cast<uint>(GET_SETTING(FileBufferSize)); }

/*! \brief Set the number of frames read at once from the file.
 *
 * \param[in] value the new number of frames read at once from the file
 */
inline void set_file_buffer_size(uint value) { UPDATE_SETTING(FileBufferSize, value); }

#pragma endregion

#pragma region File

/*! \brief Return the absolute path of the file that will be loaded.
 *
 * \return std::string the absolute file path
 */
inline std::string get_input_file_path() { return GET_SETTING(InputFilePath); }

/*! \brief Set the absolute path of the file that will be loaded.
 *
 * \param[in] value the new absolute file path
 */
inline void set_input_file_path(std::string value) { UPDATE_SETTING(InputFilePath, value); }

/*! \brief Return whether the full file will be loaded in the GPU memory or not. It's used to reduce memory transfer
 * between RAM and VRAM but it will increase GPU load.
 *
 * \return bool true if the full file will be loaded in the GPU memory
 */
inline bool get_load_file_in_gpu() { return GET_SETTING(LoadFileInGPU); }

/*! \brief Set whether the full file will be loaded in the GPU memory or not. It's used to reduce memory transfer
 * between RAM and VRAM but it will increase GPU load.
 *
 * \param[in] value true if the full file will be loaded in the GPU memory
 */
inline void set_load_file_in_gpu(bool value) { UPDATE_SETTING(LoadFileInGPU, value); }

/*! \brief Return the number of frames that will be read from the file per second.
 *
 * \return uint the input fps
 */
inline uint get_input_fps() { return static_cast<uint>(GET_SETTING(InputFPS)); }

/*! \brief Set the number of frames that will be read from the file per second.
 *
 * \param[in] value the new input fps
 */
inline void set_input_fps(uint value) { UPDATE_SETTING(InputFPS, value); }

#pragma endregion

#pragma region File Offset

/*! \brief Gets the index of the first frame that will be read in the file.
 *
 * Frames read will be in range [get_input_file_start_index, get_input_file_end_index]
 *
 * \return size_t the file start index
 */
inline size_t get_input_file_start_index() { return GET_SETTING(InputFileStartIndex); }

/*! \brief Sets the index of the first frame that will be read in the file.
 *
 * Frames read will be in range [get_input_file_start_index, get_input_file_end_index]
 *
 * \param[in] value the new first frame index
 */
void set_input_file_start_index(size_t value);

/*! \brief Gets the index of the last frame that will be read in the file.
 *
 * Frames read will be in range [get_input_file_start_index, get_input_file_end_index]
 *
 * \return size_t the file end index
 */
inline size_t get_input_file_end_index() { return GET_SETTING(InputFileEndIndex); }

/*! \brief Sets the index of the last frame that will be read in the file.
 *
 * Frames read will be in range [get_input_file_start_index, get_input_file_end_index]
 *
 * \param[in] value the new last frame index
 */
void set_input_file_end_index(size_t value);

#pragma endregion

#pragma region File Import

/*! \brief Launchs the reading of the loaded file.
 *
 * Will stop any computation currently running (camera or file loading).
 *
 * \return bool false on failure
 */
bool import_start();

/*! \brief Stops the importing of the file and all computation */
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

/*! \brief Return the name of the camera ini file used.
 *
 * \return const char* the camera ini file name
 */
inline const char* get_camera_ini_name() { return Holovibes::instance().get_camera_ini_name(); }

/*! \brief Return the kind of camera used or none if no camera is used.
 *
 * \return CameraKind the camera kind
 */
inline CameraKind get_camera_kind() { return GET_SETTING(CameraKind); }

/*! \brief Changes the current camera used or none if no camera is used. Will stop any computation currently running and
 * start the frame acquisition.
 *
 * \param[in] value the new camera kind
 * \param[in] save whether to save the camera kind in the user settings file.
 *
 *  \return bool true on success
 */
bool set_camera_kind(CameraKind value, bool save = true);

/*! \brief Return the number of frames that will be read from the camera per second. If no camera is used, it will
 * return the fps of the camera used when recording the file.
 *
 * \return uint the camera fps
 */
inline uint get_camera_fps() { return GET_SETTING(CameraFps); }

/*! \brief Set the number of frames that will be read from the camera per second.
 *
 * \param[in] value the new camera fps
 */
inline void set_camera_fps(uint value) { UPDATE_SETTING(CameraFps, value); }

/*! \brief Return the physical pixel size of the camera sensor (the one in use or the one used when recording the file)
 * in µm.
 *
 * Ex: get_pixel_size() == 5.5f means that each pixel of the camera sensor is 5.5 µm wide.
 *
 * \return float the pixel size in µm
 */
inline float get_pixel_size() { return GET_SETTING(PixelSize); }

/*! \brief Set the physical pixel size of the camera sensor (the one in use or the one used when recording the file) in
 * µm.
 *
 * Ex: set_pixel_size(5.5f) means that each pixel of the camera sensor is 5.5 µm wide.
 *
 * \param[in] value the new pixel size in µm
 */
inline void set_pixel_size(float value) { UPDATE_SETTING(PixelSize, value); }

#pragma endregion

} // namespace holovibes::api