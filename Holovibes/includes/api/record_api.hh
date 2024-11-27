/*! \file
 *
 * \brief Regroup all functions used to interact with the recording.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

inline std::string get_record_file_path() { return GET_SETTING(RecordFilePath); }
inline void set_record_file_path(std::string value) { UPDATE_SETTING(RecordFilePath, value); }

inline std::optional<size_t> get_record_frame_count() { return GET_SETTING(RecordFrameCount); }
inline void set_record_frame_count(std::optional<size_t> value) { UPDATE_SETTING(RecordFrameCount, value); }

inline RecordMode get_record_mode() { return GET_SETTING(RecordMode); }
inline void set_record_mode(RecordMode value) { UPDATE_SETTING(RecordMode, value); }

inline size_t get_record_frame_skip() { return GET_SETTING(RecordFrameSkip); }
inline void set_record_frame_skip(size_t value) { UPDATE_SETTING(RecordFrameSkip, value); }

inline bool get_frame_record_enabled() { return GET_SETTING(FrameRecordEnabled); }
inline void set_frame_record_enabled(bool value) { UPDATE_SETTING(FrameRecordEnabled, value); }

inline bool get_chart_record_enabled() { return GET_SETTING(ChartRecordEnabled); }
inline void set_chart_record_enabled(bool value) { UPDATE_SETTING(ChartRecordEnabled, value); }

inline uint get_nb_frame_skip() { return GET_SETTING(FrameSkip); }

inline uint get_mp4_fps() { return GET_SETTING(Mp4Fps); }
inline void set_mp4_fps(uint value) { UPDATE_SETTING(Mp4Fps, value); }

/*! \brief Checks preconditions to start recording
 *
 * \return success if all preconditions are met
 */
bool start_record_preconditions();

/*!
 * \brief Initiates the recording process.
 *
 * This function starts the recording process based on the current recording mode.
 * It executes the provided callback function once the recording is complete.
 *
 * \param[in] callback A lambda function to execute at the end of the recording process.
 *                 Note: The API should not handle callbacks directly. This needs to be fixed (FIXME).
 */
void start_record(std::function<void()> callback);

/*! \brief Stops recording
 *
 * \note This functions calls the notification `record_stop` when this is done.
 */
void stop_record();

/*! \brief Change the record mode in the settings
 *
 * \param[in] value The new record mode to be set to
 */
void set_record_mode_enum(RecordMode value);

/*!
 * \brief Gets the available extension for the given record mode
 *
 * \param[in] mode The record mode for which to get the available extensions
 * \return std::vector<OutputFormat> The available file extensions as an enum.
 */
std::vector<OutputFormat> get_supported_formats(RecordMode mode);

/*! \brief Return whether we are recording or not
 *
 * \return true if recording, else false
 */
bool is_recording();

/*! \brief Get the record queue location, either gpu or cpu
 *
 * \return the location of the record queue
 */
inline holovibes::Device get_record_queue_location() { return GET_SETTING(RecordQueueLocation); }

/*!
 * \brief Set the record queue location, between gpu and cpu
 *
 * \param[in] gpu whether the record queue is on the gpu or the cpu
 */
void set_record_queue_location(Device device);

/*! \brief Get the record buffer size
 *
 * \return the size of the buffer
 */
inline uint get_record_buffer_size() { return static_cast<uint>(GET_SETTING(RecordBufferSize)); }

/*! \brief Set the record buffer size, and trigger the allocation of the pipe
 *
 * \param[in] value the size of the buffer
 */
void set_record_buffer_size(uint value);

} // namespace holovibes::api