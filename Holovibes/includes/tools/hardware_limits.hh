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

/*! \file
 *
 * Getters of the GPU's specs so that calculations are optimized  */
#pragma once

/*! \brief Getter on max threads in one dimension
**
** Fetch the maximum number of threads available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads_1d();

/*! \brief Getter on max threads in two dimensions
**
** Fetch the maximum number of threads available in two dimensions
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads_2d();

/*! \brief Getter on max blocks
**
** Fetch the maximum number of blocks available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_blocks();