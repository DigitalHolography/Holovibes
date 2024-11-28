/*! \file
 *
 * \brief Regroup all functions used to interact with the Space transform and the time transform alogrithm.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

inline SpaceTransformation get_space_transformation() { return GET_SETTING(SpaceTransformation); }

inline uint get_time_stride() { return GET_SETTING(TimeStride); }
void set_time_stride(uint value);

inline uint get_batch_size() { return GET_SETTING(BatchSize); }
bool set_batch_size(uint value);

inline float get_lambda() { return GET_SETTING(Lambda); }

inline ViewPQ get_p() { return GET_SETTING(P); }
inline int get_p_accu_level() { return GET_SETTING(P).width; }
inline uint get_p_index() { return GET_SETTING(P).start; }

inline ViewPQ get_q(void) { return GET_SETTING(Q); }
inline uint get_q_index() { return GET_SETTING(Q).start; }
inline uint get_q_accu_level() { return GET_SETTING(Q).width; }

inline ViewXY get_x(void) { return GET_SETTING(X); }
inline uint get_x_cuts() { return GET_SETTING(X).start; }
inline int get_x_accu_level() { return GET_SETTING(X).width; }

inline ViewXY get_y(void) { return GET_SETTING(Y); }
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

#pragma region FFT Shift

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

#pragma endregion

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

/*!
 * \brief Modifies wave length (represented by lambda symbol in phisics)
 *
 * \param value the new value
 */
void set_lambda(float value);

inline float get_z_distance() { return GET_SETTING(ZDistance); }

/*!
 * \brief Sets the distance value for the z-coordinate.
 *
 * This function updates the internal setting for the z-coordinate distance
 * and sends a notification to the `z_distance` notifier. Additionally,
 * it refreshes the pipeline if the computation mode is not set to raw.
 *
 * \param value The new z-coordinate distance value.
 *
 * \note
 * - This function sends the notification `z_distance` with the new value when called.
 * - If the computation mode is `Computation::Raw`, the function returns immediately
 *   without updating the setting or refreshing the pipeline.
 */
void set_z_distance(float value);

/*! \brief Modifies space transform calculation
 *
 * \param value the string to match to determine the kind of space transformation
 */
void set_space_transformation(const SpaceTransformation value);

/*! \brief Modifies time transform calculation
 *
 * \param value the string to match to determine the kind of time transformation
 */
void set_time_transformation(const TimeTransformation value);

/*! \brief Enables or Disables unwrapping 2d
 *
 * \param value true: enable, false: disable
 */
void set_unwrapping_2d(const bool value);

/*! \brief Changes the time transformation size from ui value
 *
 * \param time_transformation_size The new time transformation size
 */
void update_time_transformation_size(uint time_transformation_size);

/*! \brief Modifies time transformation stride size from ui value
 *
 * \param time_stride the new value
 */
void update_time_stride(const uint time_stride);

/*! \brief Modifies batch size from ui value. Used when the image mode is changed ; in this case neither batch_size or
 * time_stride were modified on the GUI, so no notify is needed.
 *
 * \param batch_size the new value
 */
void update_batch_size(uint batch_size);

} // namespace holovibes::api