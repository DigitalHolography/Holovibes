/*! \file record_api.hh
 *
 * \brief Regroup all functions used to interact with the recording.
 */
#pragma once

#include "common_api.hh"
#include "enum_recorded_eye_type.hh"

namespace holovibes::api
{

class RecordApi : public IApi
{

  public:
    RecordApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Record File

    /*! \brief Return the absolute path of where the record will be saved.
     *
     * The file extension will determine the format of the record. To see supported formats for a particular record
     * mode, see \ref holovibes::api::get_supported_formats "this".
     *
     * \return std::string the absolute file path
     */
    inline std::string get_record_file_path() const { return GET_SETTING(RecordFilePath); }

    /*! \brief Set the absolute path of where the record will be saved.
     *
     * The file extension will determine the format of the record. To see supported formats for a particular record
     * mode, see \ref holovibes::api::get_supported_formats "this".
     *
     * \param[in] value the new absolute file path
     */
    inline void set_record_file_path(std::string value) const { UPDATE_SETTING(RecordFilePath, value); }

    /*! \brief Return the number of frames that will be recorded per second for the mp4 format.
     *
     * \return uint the number of frames per second
     */
    inline uint get_mp4_fps() const { return GET_SETTING(Mp4Fps); }

    /*! \brief Set the number of frames that will be recorded per second for the mp4 format.
     *
     * \param[in] value the new number of frames per second
     */
    inline void set_mp4_fps(uint value) const { UPDATE_SETTING(Mp4Fps, value); }

#pragma endregion

#pragma region Frame Record

    /*! \brief Return whether the frame recording is enabled or not.
     *
     * \return bool true if frame recording is enabled
     */
    inline bool get_frame_record_enabled() const { return GET_SETTING(FrameRecordEnabled); }

    /*! \brief Set whether the frame recording is enabled or not.
     *
     * \param[in] value true if frame recording is enabled
     * \warning This function is internal
     */
    inline void set_frame_record_enabled(bool value) const { UPDATE_SETTING(FrameRecordEnabled, value); }

    /*! \brief Return the number of frames that will be recorded. This number is multiplied by three in case of moments
     * recording since one frame will results in three moments.
     *
     * \return std::optional<size_t> the record frame count
     */
    inline std::optional<size_t> get_record_frame_count() const { return GET_SETTING(RecordFrameCount); }

    /*! \brief Set the number of frames that will be recorded. This number is multiplied by three in case of moments
     * recording since one frame will results in three moments.
     *
     * \param[in] value the new record frame count
     */
    inline void set_record_frame_count(std::optional<size_t> value) const { UPDATE_SETTING(RecordFrameCount, value); }

    /*! \brief Return the number of frames that will be skipped at the beginning of the recording.
     *
     * Ex: `get_record_frame_offset() == 10` means that the first 10 frames will be skipped.
     *
     * \return size_t the record frame offset
     */
    inline size_t get_record_frame_offset() const { return GET_SETTING(RecordFrameOffset); }

    /*! \brief Sets the number of frames that will be skipped at the beginning of the recording.
     *
     * Ex: `set_record_frame_offset(10)` means that the first 10 frames will be skipped.
     *
     * \param[in] value the new record frame offset
     */
    inline void set_record_frame_offset(size_t value) const { UPDATE_SETTING(RecordFrameOffset, value); }

    /*! \brief Return the number of frames that will be skipped before saving a frame.
     *
     * Ex:
     * - `get_nb_frame_skip() == 1` means that only even frames will be saved.
     * - `get_nb_frame_skip() == 2` means that only frames 0, 3, 6, 9, ... will be saved.
     * - `get_nb_frame_skip() == 3` and get_record_frame_offset() == 10 means that only frames 10, 14, 18, 22, ... will
     * be saved.
     *
     * \return uint the record frame skip
     */
    inline uint get_nb_frame_skip() const { return GET_SETTING(FrameSkip); }

    /*! \brief Sets the number of frames that will be skipped before saving a frame.
     *
     * Ex:
     * - `set_nb_frame_skip(1)` means that only even frames will be saved.
     * - `set_nb_frame_skip(2)` means that only frames 0, 3, 6, 9, ... will be saved.
     * - `set_nb_frame_skip(3)` and get_record_frame_offset() == 10 means that only frames 10, 14, 18, 22, ... will be
     * saved.
     *
     * \param[in] value the new record frame skip
     */
    inline void set_nb_frame_skip(uint value) const { UPDATE_SETTING(FrameSkip, value); }

#pragma endregion

#pragma region Chart Record

    /*! \brief Return whether the chart recording is enabled or not.
     *
     * \return bool true if chart recording is enabled
     */
    inline bool get_chart_record_enabled() const { return GET_SETTING(ChartRecordEnabled); }

    /*! \brief Set whether the chart recording is enabled or not.
     *
     * \param[in] value true if chart recording is enabled
     * \warning This function is internal
     */
    inline void set_chart_record_enabled(bool value) const { UPDATE_SETTING(ChartRecordEnabled, value); }

    /*! \brief Return the rectangular region used as the signal zone.
     *
     * \return units::RectFd the signal zone
     */
    inline units::RectFd get_signal_zone() const { return GET_SETTING(SignalZone); }

    /*! \brief Set the rectangular region used as the signal zone.
     *
     * \param[in] rect the new signal zone
     */
    inline void set_signal_zone(const units::RectFd& rect) const { UPDATE_SETTING(SignalZone, rect); }

    /*! \brief Return the rectangular region used as the noise zone.
     *
     * \return units::RectFd the noise zone
     */
    inline units::RectFd get_noise_zone() const { return GET_SETTING(NoiseZone); }

    /*! \brief Set the rectangular region used as the noise zone.
     *
     * \param[in] rect the new noise zone
     */
    inline void set_noise_zone(const units::RectFd& rect) const { UPDATE_SETTING(NoiseZone, rect); }

#pragma endregion

#pragma region Record Mode

    /*! \brief Return the record mode (raw, holgram, moments, etc.).
     *
     * \return RecordMode the record mode
     */
    inline RecordMode get_record_mode() const { return GET_SETTING(RecordMode); }

    /*! \brief Set the record mode (raw, holgram, moments, etc.).
     *
     * \param[in] value the new record mode
     * \warning This function is not intended for realtime use.
     */
    inline void set_record_mode(RecordMode value) const { UPDATE_SETTING(RecordMode, value); }

    /*! \brief Change the record mode (raw, holgram, moments, etc.).
     *
     * \param[in] value The new record mode
     * \warning This function is intended for realtime use.
     */
    void set_record_mode_enum(RecordMode value) const;

    /*!
     * \brief Gets the available extension for the given record mode
     *
     * \param[in] mode The record mode for which to get the available extensions
     * \return std::vector<OutputFormat> The available file extensions as an enum.
     */
    std::vector<OutputFormat> get_supported_formats(RecordMode mode) const;

#pragma endregion

#pragma region Eye

    /**
     * \brief Get the eye that is recorded by the program
     *
     * \return true The RIGHT eye
     * \return false The LEFT eye
     */
    inline RecordedEyeType get_recorded_eye() const { return GET_SETTING(RecordedEye); }

    /**
     * \brief Sets the eye to be recorded; this only affects how the recording is called
     *
     * \param value[in] Which eye to record
     */
    void set_recorded_eye(RecordedEyeType value) const;

    /**
     * \brief Gets a string representation of the current recorded eye
     * This string is destined to be used for file purposes.
     *
     * \return std::string The stringified recorded eye, either "L" or "R", or "" for no eye
     */
    std::string get_recorded_eye_file_string() const;

    /**
     * \brief Gets a string representation of the current recorded eye
     * This string is more explicit and should be used for display purposes.
     *
     * \return std::string The stringified recorded eye, either "L" or "R", or "" for no eye
     */
    std::string get_recorded_eye_display_string() const;

#pragma endregion

#pragma region Record

    /*! \brief Checks preconditions to start recording
     *
     * \return bool true if all preconditions are met
     */
    bool start_record_preconditions() const;

    /*!
     * \brief Initiates the recording process.
     *
     * This function starts the recording process based on the current recording mode.
     * It executes the provided callback function once the recording is complete.
     *
     * \param[in] callback A lambda function to execute at the end of the recording process.
     *                 Note: The API should not handle callbacks directly. This needs to be fixed (FIXME).
     */
    void start_record(std::function<void()> callback) const;

    /*! \brief Stops recording
     *
     * \note This functions calls the notification `record_stop` when this is done.
     */
    void stop_record() const;

    /*! \brief Return whether we are recording or not
     *
     * \return bool true if recording, else false
     */
    bool is_recording() const;

#pragma endregion

#pragma region Record Queue

    /*! \brief Gets the record queue location, either gpu or cpu
     *
     * \return Device the location of the record queue
     */
    inline holovibes::Device get_record_queue_location() const { return GET_SETTING(RecordQueueLocation); }

    /*!
     * \brief Sets the record queue location, between gpu and cpu
     *
     * \param[in] gpu whether the record queue is on the gpu or the cpu
     */
    void set_record_queue_location(Device device) const;

    /*! \brief Gets the capacity (number of frames) of the record queue.
     *
     * \return uint the size of the record queue
     */
    inline uint get_record_buffer_size() const { return static_cast<uint>(GET_SETTING(RecordBufferSize)); }

    /*! \brief Sets the capacity (number of frames) of the record queue and rebuild it.
     *
     * \param[in] value the size of the record queue
     */
    void set_record_buffer_size(uint value) const;

#pragma endregion
};

} // namespace holovibes::api