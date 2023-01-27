/*! \file
 *
 * \brief Getters of the GPU's specs so that calculations are optimized
 */
#pragma once

/*! \brief Getter on max threads in one dimension
 *
 * Fetch the maximum number of threads available in one dimension
 * for a kernel/CUDA ComputeModeEnum. It asks directly the
 * NVIDIA graphic card. This function, when called several times,
 * will only ask once the hardware.
 */
unsigned int get_max_threads_1d();

/*! \brief Getter on max threads in two dimensions
 *
 * Fetch the maximum number of threads available in two dimensions
 * for a kernel/CUDA ComputeModeEnum. It asks directly the
 * NVIDIA graphic card. This function, when called several times,
 * will only ask once the hardware.
 */
unsigned int get_max_threads_2d();

/*! \brief Getter on max blocks
 *
 * Fetch the maximum number of blocks available in one dimension
 * for a kernel/CUDA ComputeModeEnum. It asks directly the
 * NVIDIA graphic card. This function, when called several times,
 * will only ask once the hardware.
 */
unsigned int get_max_blocks();