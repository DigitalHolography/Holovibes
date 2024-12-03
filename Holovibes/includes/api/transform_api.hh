/*! \file transform_api.hh
 *
 * \brief Regroup all functions used to interact with the Space transform and the time transform alogrithm.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

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
inline uint get_batch_size() { return GET_SETTING(BatchSize); }

/*! \brief Modifies the batch size. Updates time stride if needed.
 *
 * The batch size defined:
 * - The number of frames that are processed at the same time for the space transformation.
 * - The size of the batch in the input queue
 * - The number of frames recorded at once in raw record.
 *
 * The batch size must be greater than `time_stride`.
 *
 * \param[in] value the new value
 * \warning This function is not intended for realtime use.
 *
 * \return bool true if the time stride needs to be updated
 */
bool set_batch_size(uint value);

/*! \brief Modifies the batch size. Updates time stride if needed.
 *
 * \param[in] batch_size the new value
 * \warning This function is intended for realtime use.
 */
void update_batch_size(uint batch_size);

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
inline uint get_time_stride() { return GET_SETTING(TimeStride); }

/*! \brief Modifies the time stride. Must be a multiple of `batch_size` and greater equal than `batch_size`.
 *
 * It defines the number of batch skipped before processing frames. `skipped = (time_stride` / `batch_size`) - 1`.
 * For example:
 * - `batch_size` = 10, `time_stride` = 20, 1 batch is skipped
 * - `batch_size` = 10, `time_stride` = 60, 6 batch are skipped
 * - `batch_size` = 10, `time_stride` = 10, 0 batch are skipped
 *
 * \param[in] value the new value
 * \warning This function is not intended for realtime use.
 */
void set_time_stride(uint value);

/*! \brief Modifies the time stride. Must be a multiple of `batch_size` and greater equal than `batch_size`.
 *
 * \param[in] time_stride the new value
 * \warning This function is intended for realtime use.
 */
void update_time_stride(const uint time_stride);

#pragma endregion

#pragma region Space Tr.

/*! \brief Returns the space transformation algorithm used (either Fresnel or Angular Spectrum).
 *
 * \return SpaceTransformation the space transformation algorithm
 */
inline SpaceTransformation get_space_transformation() { return GET_SETTING(SpaceTransformation); }

/*! \brief Modifies the space transformation algorithm used (either Fresnel or Angular Spectrum).
 *
 * \param value the new value
 * \warning This function is intended for realtime use.
 */
void set_space_transformation(const SpaceTransformation value);

/*! \brief Returns the wave length of the laser. // TODO(etienne): metrics
 *
 * \return float the wave length
 */
inline float get_lambda() { return GET_SETTING(Lambda); }

/*!
 * \brief Sets the wave length of the laser. // TODO(etienne): metrics
 *
 * \param[in] value the new value
 */
void set_lambda(float value);

/*! \brief Returns the distance value for the z-coordinate (the focus). // TODO(etienne): metrics
 *
 * \return float the z-coordinate distance value
 */
inline float get_z_distance() { return GET_SETTING(ZDistance); }

/*!
 * \brief Sets the distance value for the z-coordinate (the focus). // TODO(etienne): metrics
 *
 * \param value The new z-coordinate distance value.
 */
void set_z_distance(float value);

#pragma endregion

#pragma region Time Tr.

inline int get_p_accu_level() { return GET_SETTING(P).width; }
inline uint get_p_index() { return GET_SETTING(P).start; }

inline uint get_q_index() { return GET_SETTING(Q).start; }
inline uint get_q_accu_level() { return GET_SETTING(Q).width; }

inline uint get_x_cuts() { return GET_SETTING(X).start; }
inline int get_x_accu_level() { return GET_SETTING(X).width; }

inline uint get_y_cuts() { return GET_SETTING(Y).start; }
inline int get_y_accu_level() { return GET_SETTING(Y).width; }

/*!
 * \name Time transformation
 * \{
 */
inline TimeTransformation get_time_transformation() { return GET_SETTING(TimeTransformation); }

inline uint get_time_transformation_size() { return GET_SETTING(TimeTransformationSize); }
inline void set_time_transformation_size(uint value) { UPDATE_SETTING(TimeTransformationSize, value); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return GET_SETTING(TimeTransformationCutsOutputBufferSize);
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    UPDATE_SETTING(TimeTransformationCutsOutputBufferSize, value);
}
/*! \} */

/*! \brief Changes the time transformation size from ui value
 *
 * \param time_transformation_size The new time transformation size
 */
void update_time_transformation_size(uint time_transformation_size);

/*! \brief Modifies p accumulation
 *
 * \param p_value the new value of p accu
 */
void set_p_accu_level(uint p_value);

/*! \brief Modifies x accumulation
 *
 * \param x_value the new value of x accu
 */
void set_x_accu_level(uint x_value);

/*! \brief Modifies x cuts
 *
 * \param x_value the new value of x cuts
 */
void set_x_cuts(uint x_value);

/*! \brief Modifies y accumulation
 *
 * \param y_value the new value of y accu
 */
void set_y_accu_level(uint y_value);

/*! \brief Modifies y cuts
 *
 * \param y_value the new value of y cuts
 */
void set_y_cuts(uint y_value);

/*! \brief Modifies q accumulation
 *
 * \param is_q_accu if q accumulation is allowed
 * \param q_value the new value of q accu
 */
void set_q_accu_level(uint q_value);

/*! \brief Modifies x and y
 *
 * \param x value to modify
 * \param y value to modify
 */
void set_x_y(uint x, uint y);

/*! \brief Modifies p
 *
 * \param value the new value of p
 */
void set_p_index(uint value);

/*! \brief Modifies q
 *
 * \param value the new value of q
 */
void set_q_index(uint value);

/*! \brief Limit the value of p_index and p_acc according to time_transformation_size */
void check_p_limits();

/*! \brief Limit the value of q_index and q_acc according to time_transformation_size */
void check_q_limits();

/*! \brief Increment p by 1 */
void increment_p();

/*! \brief Decrement p by 1 */
void decrement_p();

/*! \brief Modifies time transform calculation
 *
 * \param value the string to match to determine the kind of time transformation
 */
void set_time_transformation(const TimeTransformation value);

#pragma endregion

#pragma region Specials

/*!
 * \name FFT
 * \{
 */
/*! \brief Getter and Setter for the fft shift, triggered when FFT Shift button is clicked on the gui. (Setter refreshes
 * the pipe) */
inline bool get_fft_shift_enabled() { return GET_SETTING(FftShiftEnabled); }
inline bool get_registration_enabled();
void set_fft_shift_enabled(bool value);
/*! \} */

/*! \brief Enables or Disables unwrapping 2d
 *
 * \param value true: enable, false: disable
 */
void set_unwrapping_2d(const bool value);

#pragma endregion

} // namespace holovibes::api