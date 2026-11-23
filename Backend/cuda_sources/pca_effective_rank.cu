#include "pca_effective_rank.cuh"

namespace holovibes::compute
{
namespace
{
__global__ void remove_effective_rank(cuComplex* vectors,
                                      const float* eigenvalues,
                                      unsigned frames,
                                      const int* solver_info,
                                      float rank_factor,
                                      int* batch_error)
{
    // One CUDA block per eigensystem. cuSOLVER puts the strongest modes last.
    const size_t size = static_cast<size_t>(frames) * frames;
    vectors += blockIdx.x * size;
    eigenvalues += static_cast<size_t>(blockIdx.x) * frames;
    __shared__ unsigned removed;
    if (threadIdx.x == 0)
    {
        double total = 0.;
        int error = solver_info[blockIdx.x];
        for (unsigned i = 0; i < frames && !error; ++i)
        {
            if (!isfinite(eigenvalues[i]))
                error = -1;
            // Negative eigenvalues from roundoff carry no probability.
            total += fmax(0., static_cast<double>(eigenvalues[i]));
        }

        removed = 0;
        if (error)
        {
            if (batch_error)
                atomicCAS(batch_error, 0, error);
            removed = frames;
        }
        else if (total > 0. && rank_factor > 0.f)
        {
            double entropy = 0.;
            for (unsigned i = 0; i < frames; ++i)
            {
                const double p = fmax(0., static_cast<double>(eigenvalues[i])) / total;
                if (p > 0.)
                    entropy -= p * log(p);
            }
            // Multiply before rounding and tolerate roundoff near integer ranks.
            const double scaled_rank = static_cast<double>(rank_factor) * exp(entropy);
            removed = static_cast<unsigned>(fmax(1., fmin(static_cast<double>(frames), ceil(scaled_rank - 1e-6))));
        }
    }
    __syncthreads();

    const size_t first_removed = static_cast<size_t>(frames - removed) * frames;
    for (size_t i = first_removed + threadIdx.x; i < size; i += blockDim.x)
        vectors[i] = make_cuComplex(0.f, 0.f);
}
} // namespace

cudaError_t remove_effective_rank_components(cuComplex* eigenvectors,
                                             const float* eigenvalues,
                                             unsigned frames,
                                             const int* solver_info,
                                             float rank_factor,
                                             cudaStream_t stream,
                                             unsigned batch_size,
                                             int* batch_error)
{
    remove_effective_rank<<<batch_size, 256, 0, stream>>>(eigenvectors,
                                                          eigenvalues,
                                                          frames,
                                                          solver_info,
                                                          rank_factor,
                                                          batch_error);
    return cudaGetLastError();
}
} // namespace holovibes::compute
