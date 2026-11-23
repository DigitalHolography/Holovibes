#pragma once

#include <cuComplex.h>
#include <cuda_runtime_api.h>

namespace holovibes::compute
{
/*! \brief Zero the strongest columns of each eigenvector matrix using its scaled entropy effective rank.
 *
 * Eigenvalues are sorted in ascending order and matrices are column-major.
 * If supplied, batch_error records the first eigensolver failure or nonfinite spectrum.
 */
cudaError_t remove_effective_rank_components(cuComplex* eigenvectors,
                                             const float* eigenvalues,
                                             unsigned frames,
                                             const int* solver_info,
                                             float rank_factor,
                                             cudaStream_t stream,
                                             unsigned batch_size = 1,
                                             int* batch_error = nullptr);
} // namespace holovibes::compute
