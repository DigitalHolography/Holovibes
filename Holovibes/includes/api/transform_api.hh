/*! \file transform_api.hh
 *
 * \brief Regroup all functions used to interact with the Space transform and the time transform alogrithm.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

class TransformApi : public IApi
{

  public:
    TransformApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Batch

    /*! \brief Returns the batch size.
     *
     * The batch size defined:
     * - The number of frames that are processed at the same time for the space transformation.
     * - The size of the batch in the input queue
     * - The number of frames recorded at once in raw record.
     *
     * The batch size must be greater than `time_stride`.
     *
     * \return uint the batch size
     */
    inline uint get_batch_size() const { return GET_SETTING(BatchSize); }

    /*! \brief Modifies the batch size. Updates time stride if needed.
     *
     * The batch size defined:
     * - The number of frames that are processed at the same time for the space transformation.
     * - The size of the batch in the input queue
     * - The number of frames recorded at once in raw record.
     *
     * The batch size must be greater than `time_stride`.
     *
     * \param[in] batch_size the new value
     *
     * \return ApiCode the status of the modification: OK or NO_CHANGE.
     */
    ApiCode set_batch_size(uint batch_size) const;

#pragma endregion

#pragma region Time Stride

    /*! \brief Returns the time stride.
     *
     * It defines the number of batch skipped before processing frames. It's a multiple of `batch_size` and it's greater
     * equal than `batch_size`.`skipped = (time_stride` / `batch_size`) - 1`.
     * For example:
     * - `batch_size` = 10, `time_stride` = 20, 1 batch is skipped
     * - `batch_size` = 10, `time_stride` = 60, 6 batch are skipped
     * - `batch_size` = 10, `time_stride` = 10, 0 batch are skipped
     *
     * \return uint the time stride
     */
    inline uint get_time_stride() const { return GET_SETTING(TimeStride); }

    /*! \brief Modifies the time stride. Must be a multiple of `batch_size` and greater equal than `batch_size`.
     *
     * It defines the number of batch skipped before processing frames. `skipped = (time_stride` / `batch_size`) - 1`.
     * For example:
     * - `batch_size` = 10, `time_stride` = 20, 1 batch is skipped
     * - `batch_size` = 10, `time_stride` = 60, 6 batch are skipped
     * - `batch_size` = 10, `time_stride` = 10, 0 batch are skipped
     *
     * \param[in] time_stride the new value
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_time_stride(uint time_stride) const;

#pragma endregion

#pragma region Space Tr.

    /*! \brief Returns the space transformation algorithm used (either Fresnel or Angular Spectrum).
     *
     * \return SpaceTransformation the space transformation algorithm
     */
    inline SpaceTransformation get_space_transformation() const { return GET_SETTING(SpaceTransformation); }

    /*! \brief Modifies the space transformation algorithm used (either Fresnel or Angular Spectrum).
     *
     * \param[in] value the new value
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_space_transformation(const SpaceTransformation value) const;

    /*! \brief Returns the wave length of the laser (in nm).
     *
     * \return float the wave length (in nm)
     */
    inline float get_lambda() const { return GET_SETTING(Lambda); }

    /*!
     * \brief Sets the wave length of the laser (in nm). Must be positive.
     *
     * \param[in] value the new value (in nm)
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_lambda(float value) const;

    /*! \brief Returns the distance in meter for the z-coordinate (the focus).
     *
     * \return float the z-coordinate distance in meter.
     */
    inline float get_z_distance() const { return GET_SETTING(ZDistance); }

    /*!
     * \brief Sets the distance in meter for the z-coordinate (the focus).
     *
     * \param[in] value The new z-coordinate distance in meter.
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_z_distance(float value) const;

#pragma endregion

#pragma region Time Tr.

    /*! \brief Returns the time transformation size. It's the number of frames used for one time transformation. Is
     * greater than 0.
     *
     * \return uint the time transformation size
     */
    inline uint get_time_transformation_size() const { return GET_SETTING(TimeTransformationSize); }

    /*! \brief Modifies the time transformation size. It's the number of frames used for one time transformation. Must
     * be greater than 0.
     *
     * \param[in] time_transformation_size the new value
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_time_transformation_size(uint time_transformation_size) const;

    /*! \brief Returns the time transformation algorithm used (STFT, PAC, etc.).
     *
     * \return TimeTransformation the time transformation algorithm
     */
    inline TimeTransformation get_time_transformation() const { return GET_SETTING(TimeTransformation); }

    /*! \brief Sets the time transformation algorithm used (STFT, PAC, etc.).
     *
     * \param[in] value the new value
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_time_transformation(const TimeTransformation value) const;

#pragma endregion

#pragma region Time Tr.Freq

    /*! \brief Returns the min accumulation frequency for time transformation. Is in range [0, `time_transformation_size
     * - get_p_accu_level - 1`].
     *
     * After the time transformation, the frequency ranging (resulting from the FFT) between [`get_p_index`,
     * `get_p_index + get_p_accu_level`] will be accumulated into one image.
     *
     * \return uint the min accumulation frequency
     */
    inline uint get_p_index() const { return GET_SETTING(P).start; }

    /*! \brief Sets the min accumulation frequency for time transformation. Must be in range [0,
     * `time_transformation_size - get_p_accu_level - 1`].
     *
     * After the time transformation, the frequency ranging (resulting from the FFT) between [`get_p_index`,
     * `get_p_index + get_p_accu_level`] will be accumulated into one image.
     *
     * \param[in] value the new min accumulation frequency
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_p_index(uint value) const;

    /*! \brief Returns the number of frequencies accumulated for the time transformation. Is in range [0,
     * `time_transformation_size - get_p_index - 1`].
     *
     * After the time transformation, the frequency ranging (resulting from the FFT) between [`get_p_index`,
     * `get_p_index + get_p_accu_level`] will be accumulated into one image.
     *
     * \return uint the number of frequencies accumulated
     */
    inline uint get_p_accu_level() const { return GET_SETTING(P).width; }

    /*! \brief Sets the number of frequencies accumulated for the time transformation. Must be in range [0,
     * `time_transformation_size - get_p_index - 1`].
     *
     * After the time transformation, the frequency ranging (resulting from the FFT) between [`get_p_index`,
     * `get_p_index + get_p_accu_level`] will be accumulated into one image.
     *
     * \param[in] p_value the new number of frequencies accumulated
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_p_accu_level(uint p_value) const;

    /*! \brief Returns the min eigen value index kept by the SVD. Is in range [0, `time_transformation_size -
     * get_p_accu_level - 1`].
     *
     * Only eigen values ranging between [`get_q_index`, `get_q_index + get_q_accu_level`] will be ketp.
     *
     * \return uint the new min eigen value index kept
     */
    inline uint get_q_index() const { return GET_SETTING(Q).start; }

    /*! \brief Sets the min eigen value index kept by the SVD. Must be in range [0, `time_transformation_size -
     * get_q_accu_level - 1`].
     *
     * Only eigen values ranging between [`get_q_index`, `get_q_index + get_q_accu_level`] will be ketp.
     *
     * \param[in] value the new min eigen value index kept
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_q_index(uint value) const;

    /*! \brief Returns the number of eigen values kept by the SVD. Is in range [0, `time_transformation_size -
     * get_q_index - 1`].
     *
     * Only eigen values ranging between [`get_q_index`, `get_q_index + get_q_accu_level`] will be ketp.
     *
     * \return uint the number of eigen values kept
     */
    inline uint get_q_accu_level() const { return GET_SETTING(Q).width; }

    /*! \brief Sets the number of eigen values kept by the SVD. Must be in range [0, `time_transformation_size -
     * get_q_index
     * - 1`].
     *
     * Only eigen values ranging between [`get_q_index`, `get_q_index + get_q_accu_level`] will be ketp.
     *
     * \param[in] value the new number of eigen values kept
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_q_accu_level(uint value) const;

#pragma endregion

#pragma region Time Tr.Cuts

    /*! \brief Checks the limits of the x index and x accu level.
     *
     * The x index must be in range [0, `fd.width - get_x_accu_level - 1`].
     * The x accu level must be in range [0, `fd.width - get_x_cuts - 1`].
     */
    void check_x_limits() const;

    /*! \brief Returns the start index for the x cut accumulation. Is in range [0, `fd.width - get_x_accu_level - 1`].
     *
     * \return uint the x cut start index
     */
    inline uint get_x_cuts() const { return GET_SETTING(X).start; }

    /*! \brief Sets the start index for the x cut accumulation. Must be in range [0, `fd.width - get_x_accu_level - 1`].
     *
     * \param[in] x_value the new x cut start index.
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_x_cuts(uint x_value) const;

    /*! \brief Returns the x cut accumulation level. Is in range [0, `time_transformation_size - get_x_cuts - 1`].
     *
     * \return uint the x cut accumulation level
     */
    inline uint get_x_accu_level() const { return GET_SETTING(X).width; }

    /*! \brief Sets the x cut accumulation level. Must be in range [0, `time_transformation_size - get_x_cuts - 1`].
     *
     * \param[in] x_value the new x cut accumulation level
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_x_accu_level(uint x_value) const;

    /*! \brief Checks the limits of the y index and y accu level.
     *
     * The y index must be in range [0, `fd.height - get_y_accu_level - 1`].
     * The y accu level must be in range [0, `fd.height - get_y_cuts - 1`].
     */
    void check_y_limits() const;

    /*! \brief Returns the start index for the y cut accumulation. Is in range [0, `fd.height - get_y_accu_level - 1`].
     *
     * \return uint the y cut start index
     */
    inline uint get_y_cuts() const { return GET_SETTING(Y).start; }

    /*! \brief Sets the start index for the y cut accumulation. Must be in range [0, `fd.height - get_y_accu_level -
     * 1`].
     *
     * \param[in] y_value the new y cut start index
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_y_cuts(uint y_value) const;

    /*! \brief Returns the y cut accumulation level. Is in range [0, `time_transformation_size - get_y_cuts - 1`].
     *
     * \return uint the y cut accumulation level
     */
    inline uint get_y_accu_level() const { return GET_SETTING(Y).width; }

    /*! \brief Sets the y cut accumulation level. Must be in range [0, `time_transformation_size - get_y_cuts - 1`].
     *
     * \param[in] y_value the new y cut accumulation level
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_y_accu_level(uint y_value) const;

    /*! \brief Returns the capacity (in number of frames) of the output buffers containing the result of the time
     * transformation cuts.
     *
     * \return uint the capacity of the output buffer
     */
    inline uint get_time_transformation_cuts_output_buffer_size() const
    {
        return GET_SETTING(TimeTransformationCutsOutputBufferSize);
    }

    /*! \brief Sets the capacity (in number of frames) of the output buffers containing the result of the time
     * transformation cuts.
     *
     * \param[in] value the new capacity
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_time_transformation_cuts_output_buffer_size(uint value) const;

#pragma endregion

#pragma region Specials

    /*! \brief Returns the fft shift status.
     *
     * After the time transform, frequencies will be ordered from 0 to N/2 - 1 and from -N/2 to -1. The fft shift will
     * reorder them the be from -N/2 to N/2. Where N is the number of frequencies (`time_transformation_size`).
     *
     * \return bool true: enabled, false: disabled
     */
    inline bool get_fft_shift_enabled() const { return GET_SETTING(FftShiftEnabled); }

    /*! \brief Enables or Disables the fft shift
     *
     * After the time transform, frequencies will be ordered from 0 to N/2 - 1 and from -N/2 to -1. The fft shift will
     * reorder them to be from -N/2 to N/2. Where N is the number of frequencies (`time_transformation_size`).
     *
     * \param[in] value true: enable, false: disable
     * \return ApiCode the status of the modification: OK, NO_CHANGE or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_fft_shift_enabled(const bool value) const;

    /*! \brief Returns the unwrapping 2d status.
     *
     * \return bool true: enabled, false: disabled
     */
    inline bool get_unwrapping_2d() const { return GET_SETTING(Unwrap2d); }

    /*! \brief Enables or Disables unwrapping 2d
     *
     * \param[in] value true: enable, false: disable
     * \return ApiCode the status of the modification: OK, NO_CHANGE, NOT_STARTED or WRONG_COMP_MODE (if in raw mode).
     */
    ApiCode set_unwrapping_2d(const bool value) const;

#pragma endregion

  private:
    /*! \brief Adjust the value of `p_index` and `p_accu_level` according to `time_transformation_size` */
    void check_p_limits() const;

    /*! \brief Adjust the value of `q_index` and `q_accu_level` according to `time_transformation_size` */
    void check_q_limits() const;
};

} // namespace holovibes::api