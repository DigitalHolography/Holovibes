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

/*! \brief Map input to output throughout a mapping function */
template <typename I, typename O, typename FUNC>
void map_generic(const I* const input,
                 O* const output,
                 const uint size,
                 const FUNC func,
                 cudaStream_t stream = 0);

/*! \brief Apply log10 on every pixel of the input */
void map_log10(float* const input,
               const uint	size,
               cudaStream_t	stream = 0);

/*! \brief Divide every pixel by value */
void map_divide(float* const input,
                const uint   size,
                const float  value,
                cudaStream_t stream = 0);

/*! \brief Multiply every pixel by value */
void map_multiply(float* const input,
                const uint   size,
                const float  value,
                cudaStream_t stream = 0);

template <typename I, typename O, typename FUNC>
void map_generic(const I* const input,
                 O* const output,
                 const uint size,
                 const FUNC func,
                 cudaStream_t stream);

#include "map.cuhxx"