/*! \file record_api.hh
 *
 * \brief Regroup all functions used to interact with the recording.
 */
#pragma once

#include "common_api.hh"
#include "enum_recorded_eye_type.hh"

namespace holovibes::api
{

struct RecordProgress
{
    uint acquired_frames;
    uint saved_frames;
    uint total_frames;
};

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
     *
     * \param[in] value the new absolute file path
     * \note If a record is already in progress, the new path will be used for the next record.
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
     * \note If a record is already in progress, the new fps will be used for the next record.
     */
    inline void set_mp4_fps(uint value) const { UPDATE_SETTING(Mp4Fps, value); }

#pragma endregion

#pragma region Frame Record

    /*! \brief Return whether the frame are being acquired or not.
     *
     * \return bool true if frame acquisition is enabled
     */
    inline bool get_frame_acquisition_enabled() const { return GET_SETTING(FrameAcquisitionEnabled); }

    /*! \brief Set whether the frame are acquired or not.
     *
     * \param[in] value true if frame acquisition is enabled
     * \warning This function must not be called outside of the backend
     */
    inline void set_frame_acquisition_enabled(bool value) const { UPDATE_SETTING(FrameAcquisitionEnabled, value); }

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
     * \note If a record is already in progress, the new count will be used for the next record.
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
     * \note If a record is already in progress, the new offset will be used for the next record.
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
     * \note If a record is already in progress, the new skip will be used for the next record.
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
     * \warning This function must not be called outside of the backend
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
     * \note The change will take effect immediately so during a recording, the new zone will be used for the next
     * frame.
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
     * \note The change will take effect immediately so during a recording, the new zone will be used for the next
     * frame.
     */
    inline void set_noise_zone(const units::RectFd& rect) const { UPDATE_SETTING(NoiseZone, rect); }

#pragma endregion

#pragma region Record Mode

    /*! \brief Return the record mode (raw, holgram, moments, etc.).
     *
     * \return RecordMode the record mode
     */
    inline RecordMode get_record_mode() const { return GET_SETTING(RecordMode); }

    /*! \brief Change the record mode (raw, holgram, moments, etc.).
     *
     * \param[in] value The new record mode
     *
     * \return ApiCode the status of the operation. NO_CHANGE if the mode is already set to the new value, OK otherwise.
     */
    ApiCode set_record_mode(RecordMode value) const;

    /*!
     * \brief Gets the available extension for the given record mode
     *
     * \param[in] mode The record mode for which to get the available extensions
     * \return std::vector<OutputFormat> The available file extensions as an enum.
     */
    std::vector<OutputFormat> get_supported_formats(RecordMode mode) const;

#pragma endregion

#pragma region Eye

    /*!
     * \brief Get the eye that is recorded by the program. This only affects the name of the file.
     *
     * \return RecordedEyeType Which eye is being recorded. Can be LEFT, RIGHT or NONE if no eye is selected
     */
    inline RecordedEyeType get_recorded_eye() const { return GET_SETTING(RecordedEye); }

    /*!
     * \brief Sets the eye to be recorded. This only affects the name of the file.
     *
     * \param[in] value Which eye to record
     * \note If a recording is already in progress, the new eye will be used for the next record.
     * \warning If the eye is marked as selected (there was eye data in the footer of the loaded file),
     * this function will do nothing
     */
    ApiCode set_recorded_eye(RecordedEyeType value) const;

    /*!
     * \brief Gets whether or not an eye had been specified in a previous recording for the current data
     *
     * \return bool true if the eye is specified, false if not
     */
    inline bool get_is_eye_selected() const { return GET_SETTING(IsEyeSelected); }

    /*!
     * \brief Sets whether or not the recorded eye had been specified in the header of the loaded file
     * Should not be used by the user, but is still available for conveniency
     *
     * \param selected Whether to mark the recorded eye as selected or not
     */
    inline void set_is_eye_selected(bool selected) const { UPDATE_SETTING(IsEyeSelected, selected); }

#pragma endregion

#pragma region Record

    /*! \brief Gets the current progress of the recording. It includes the number of frames acquired, saved and the
     * total number of frames to record.
     *
     * \return RecordProgress the current progress of the recording
     */
    RecordProgress get_record_progress() const;

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
     * \return ApiCode the status of the operation. FAILURE if the recording could not be started, OK otherwise.
     */
    ApiCode start_record(std::function<void()> callback) const;

    /*! \brief Stops recording.
     *
     * \return ApiCode the status of the operation. NOT_STARTED if no recording is in progress, OK otherwise.
     */
    ApiCode stop_record() const;

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
     * \return ApiCode the status of the operation. NO_CHANGE if the location is already set to the new value, OK
     * otherwise.
     */
    ApiCode set_record_queue_location(Device device) const;

    /*! \brief Gets the capacity (number of frames) of the record queue.
     *
     * \return uint the size of the record queue
     */
    inline uint get_record_buffer_size() const { return static_cast<uint>(GET_SETTING(RecordBufferSize)); }

    /*! \brief Sets the capacity (number of frames) of the record queue and rebuild it.
     *
     * \param[in] value the size of the record queue
     * \return ApiCode the status of the operation. NO_CHANGE if the size is already set to the new value, OK otherwise.
     */
    ApiCode set_record_buffer_size(uint value) const;

#pragma endregion
};

} // namespace holovibes::api