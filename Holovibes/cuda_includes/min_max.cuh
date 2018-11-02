/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

/*
* \brief This function destroys "frame" by doing reductions.
* \param d_frame the image
* \param h_memory_space_sdata space to store results from blocks
* \return maximum
*/
float get_maximum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res);

/*
* \brief This function destroys "frame" by doing reductions.
* \param d_frame the image
* \param h_memory_space_sdata space to store results from blocks
* \return minimum
*/
float get_minimum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res);

/*
** \brief Writes in min ptr and max ptr the extremums values of frame.
*/
void get_minimum_maximum_in_image(const float *frame, const unsigned frame_res, float* min, float* max);