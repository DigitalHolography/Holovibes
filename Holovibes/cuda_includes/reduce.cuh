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

using uint = unsigned int;

/*! \brief Reduce operation gpu side
*
* \param input Input buffer
* \param result Result of the reduce (even with double, imprecision may arise)
* \param size Input size
* \param reduce_op Operator used for the reduction
* \param atomic_op Atomic operator used for the write back in result
* \param identity_elt Identity element needed to initilize data (add needs 0, min needs max...)
*
* This kernel has been highly tuned in order to maximize the memory bandwidth usage
* Numerous benches have been done to achieve the best result possible
* Don't modify this kernel unless making benches
*/
template <typename T, typename U, typename R_OP, typename A_OP>
void reduce_generic(const T* const __restrict__ input,
                    U* const __restrict__ result,
                    const uint size,
                    const R_OP reduce_op,
                    const A_OP atomic_op,
                    const T identity_elt,
                    cudaStream_t stream = 0);

/*! \brief Reduce add operation
*
* Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
*/
template <typename T, typename U>
void reduce_add(const T* const input, U* const result, const uint size, cudaStream_t stream = 0);

// Min / Max : ushort not supported by CUDA (because of atomic operation)

/*! \brief Reduce min operation */
template <typename T>
void reduce_min(const T* const input, T* const result, const uint size, cudaStream_t stream = 0);

/*! \brief Reduce max operation */
template <typename T>
void reduce_max(const T* const input, T* const result, const uint size, cudaStream_t stream = 0);

#include "reduce.cuhxx"