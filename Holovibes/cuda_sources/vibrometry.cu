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

#include "vibrometry.cuh"

static __global__
void kernel_frame_ratio(const cuComplex	*frame_p,
						const cuComplex	*frame_q,
						cuComplex		*output,
						const uint		size)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;

  //while (index < size)
  {
    /* frame_p: a + ib */
    const float a = frame_p[index].x;
    const float b = frame_p[index].y;

    /* frame_q: c + id */
    const float c = frame_q[index].x;
    const float d = frame_q[index].y;

    const float q_squared_modulus = c * c + d * d + FLT_EPSILON;

    output[index].x = (a * c + b * d) / q_squared_modulus;
    output[index].y = (b * c - a * d) / q_squared_modulus;

   // index += blockDim.x * gridDim.x;
  }
}

void frame_ratio(const cuComplex	*frame_p,
				const cuComplex		*frame_q,
				cuComplex			*output,
				const uint			size,
				cudaStream_t		stream)
{
  uint threads = get_max_threads_1d();
  uint blocks = map_blocks_to_problem(size, threads);

  kernel_frame_ratio <<<blocks, threads, 0, stream>>>(frame_p, frame_q, output, size);
  cudaCheckError();
}
