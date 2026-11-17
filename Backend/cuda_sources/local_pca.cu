#include "local_pca.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace
{
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Local PCA CUDA error: ") + cudaGetErrorString(status));
}
void check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Local PCA cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
void check(cusolverStatus_t status)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("Local PCA cuSOLVER error: " + std::to_string(static_cast<int>(status)));
}

template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;
    ~DeviceBuffer() { cudaFree(ptr); }
    void allocate(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))); }
};

struct BlasHandle
{
    cublasHandle_t value = nullptr;
    BlasHandle() { check(cublasCreate(&value)); }
    ~BlasHandle() { cublasDestroy(value); }
};
struct SolverHandle
{
    cusolverDnHandle_t value = nullptr;
    SolverHandle() { check(cusolverDnCreate(&value)); }
    ~SolverHandle() { cusolverDnDestroy(value); }
};

unsigned grid_size(size_t count) { return static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535)); }

// Gather one width_x * width_y * frames tile. Modulo indexing implements periodic
// spatial borders, including corners and tiles larger than the image itself.
__global__ void gather_tile(const cuComplex* input,
                            cuComplex* tile,
                            size_t count,
                            unsigned width,
                            unsigned height,
                            unsigned width_x,
                            size_t tile_pixels,
                            unsigned origin_x,
                            unsigned origin_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t local = i % tile_pixels;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        tile[i] = input[(i / tile_pixels) * pixels + static_cast<size_t>(y) * width + x];
    }
}

// cuSOLVER orders the eigenvalues increasingly. Compute the Shannon effective
// rank of the Gram matrix, r = ceil(exp(-sum(p * log(p)))), p = lambda/sum(lambda),
// then zero the LAST r eigenvectors to remove the strongest components. A zero
// spectrum has rank zero; negative roundoff eigenvalues carry no probability.
__global__ void
remove_effective_rank(cuComplex* vectors, const float* eigenvalues, unsigned frames, const int* solver_info, int* error)
{
    __shared__ unsigned removed;
    if (threadIdx.x == 0)
    {
        removed = 0;
        double total = 0.;
        for (unsigned i = 0; i < frames; ++i)
        {
            if (!isfinite(eigenvalues[i]))
                atomicCAS(error, 0, -1);
            total += fmax(0., static_cast<double>(eigenvalues[i]));
        }
        if (*solver_info != 0)
            atomicCAS(error, 0, *solver_info);
        if (total > 0.)
        {
            double entropy = 0.;
            for (unsigned i = 0; i < frames; ++i)
            {
                const double p = fmax(0., static_cast<double>(eigenvalues[i])) / total;
                if (p > 0.)
                    entropy -= p * log(p);
            }
            // Avoid rounding a numerically perturbed integer rank up by one.
            removed = static_cast<unsigned>(fmax(1., fmin(static_cast<double>(frames), ceil(exp(entropy) - 1e-6))));
        }
    }
    __syncthreads();

    const size_t first_removed = static_cast<size_t>(frames - removed) * frames;
    const size_t size = static_cast<size_t>(frames) * frames;
    for (size_t i = first_removed + threadIdx.x; i < size; i += blockDim.x)
        vectors[i] = make_cuComplex(0.f, 0.f);
}

// Scatter residuals to their original pixel/time coordinates. Count every spatial
// contribution once (at t=0). The count can vary at wrapped borders when an image
// dimension is not divisible by the stride, so a fixed overlap factor is incorrect.
__global__ void accumulate_tile(const cuComplex* tile,
                                cuComplex* output,
                                unsigned* coverage,
                                size_t count,
                                unsigned width,
                                unsigned height,
                                unsigned width_x,
                                size_t tile_pixels,
                                unsigned origin_x,
                                unsigned origin_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t local = i % tile_pixels;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        const size_t pixel = static_cast<size_t>(y) * width + x;
        const size_t target = (i / tile_pixels) * pixels + pixel;
        atomicAdd(&output[target].x, tile[i].x);
        atomicAdd(&output[target].y, tile[i].y);
        if (i < tile_pixels)
            atomicAdd(coverage + pixel, 1u);
    }
}

__global__ void average_overlaps(cuComplex* output, const unsigned* coverage, size_t pixels, size_t count)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const unsigned weight = coverage[i % pixels];
        output[i] = weight ? make_cuComplex(output[i].x / weight, output[i].y / weight) : make_cuComplex(0.f, 0.f);
    }
}
} // namespace

namespace holovibes::compute
{
struct LocalPca::Impl
{
    unsigned width, height, frames, width_x, width_y, stride_x, stride_y;
    size_t pixels, tile_pixels;
    cudaStream_t stream;
    BlasHandle blas;
    SolverHandle solver;
    DeviceBuffer<cuComplex> tile, scores, covariance, workspace;
    DeviceBuffer<float> eigenvalues;
    DeviceBuffer<unsigned> coverage;
    DeviceBuffer<int> solver_info, error;
    int workspace_size = 0;

    Impl(unsigned w, unsigned h, unsigned n, unsigned wx, unsigned wy, unsigned sx, unsigned sy, cudaStream_t s)
        : width(w)
        , height(h)
        , frames(n)
        , width_x(wx)
        , width_y(wy)
        , stride_x(sx)
        , stride_y(sy)
        , pixels(static_cast<size_t>(w) * h)
        , tile_pixels(static_cast<size_t>(wx) * wy)
        , stream(s)
    {
        if (!w || !h || !n || n > 65535 || !wx || !wy || wx > 65535 || wy > 65535 || !sx || !sy || sx > wx || sy > wy ||
            tile_pixels > std::numeric_limits<int>::max())
            throw std::invalid_argument("Local PCA needs positive dimensions, 1..65535 frames/tile sides, "
                                        "and strides no larger than their tile sides");

        check(cublasSetStream(blas.value, stream));
        check(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH));
        check(cusolverDnSetStream(solver.value, stream));

        // Reuse one tile's scratch storage for the whole image. Spatial overlap
        // never requires duplicating the complete input cube for every tile.
        tile.allocate(tile_pixels * frames);
        scores.allocate(tile_pixels * frames);
        covariance.allocate(static_cast<size_t>(frames) * frames);
        eigenvalues.allocate(frames);
        coverage.allocate(pixels);
        solver_info.allocate(1);
        error.allocate(1);
        check(cusolverDnCheevd_bufferSize(solver.value,
                                          CUSOLVER_EIG_MODE_VECTOR,
                                          CUBLAS_FILL_MODE_LOWER,
                                          frames,
                                          covariance.ptr,
                                          frames,
                                          eigenvalues.ptr,
                                          &workspace_size));
        workspace.allocate(workspace_size);
    }
};

LocalPca::LocalPca(unsigned width,
                   unsigned height,
                   unsigned frames,
                   unsigned width_x,
                   unsigned width_y,
                   unsigned stride_x,
                   unsigned stride_y,
                   cudaStream_t stream)
    : impl_(std::make_unique<Impl>(width, height, frames, width_x, width_y, stride_x, stride_y, stream))
{
}
LocalPca::~LocalPca() = default;

void LocalPca::process(const cuComplex* input, cuComplex* output)
{
    if (!input || !output || input == output)
        throw std::invalid_argument("Local PCA requires distinct, nonnull input and output buffers");
    auto& a = *impl_;
    const cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    const size_t tile_count = a.tile_pixels * a.frames, count = a.pixels * a.frames;
    const int m = static_cast<int>(a.tile_pixels), n = static_cast<int>(a.frames);
    check(cudaMemsetAsync(output, 0, count * sizeof(cuComplex), a.stream));
    check(cudaMemsetAsync(a.coverage.ptr, 0, a.pixels * sizeof(unsigned), a.stream));
    check(cudaMemsetAsync(a.error.ptr, 0, sizeof(int), a.stream));

    for (unsigned y = 0; y < a.height; y += a.stride_y)
        for (unsigned x = 0; x < a.width; x += a.stride_x)
        {
            gather_tile<<<grid_size(tile_count), 256, 0, a.stream>>>(input,
                                                                     a.tile.ptr,
                                                                     tile_count,
                                                                     a.width,
                                                                     a.height,
                                                                     a.width_x,
                                                                     a.tile_pixels,
                                                                     x,
                                                                     y);
            check(cudaGetLastError());

            // Match classic PCA's uncentered complex covariance: C = H^H H.
            // Scaling C by the tile area would not change normalized eigenvalues.
            check(cublasCgemm(a.blas.value,
                              CUBLAS_OP_C,
                              CUBLAS_OP_N,
                              n,
                              n,
                              m,
                              &one,
                              a.tile.ptr,
                              m,
                              a.tile.ptr,
                              m,
                              &zero,
                              a.covariance.ptr,
                              n));
            check(cusolverDnCheevd(a.solver.value,
                                   CUSOLVER_EIG_MODE_VECTOR,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   a.covariance.ptr,
                                   n,
                                   a.eigenvalues.ptr,
                                   a.workspace.ptr,
                                   a.workspace_size,
                                   a.solver_info.ptr));
            remove_effective_rank<<<1, 256, 0, a.stream>>>(a.covariance.ptr,
                                                           a.eigenvalues.ptr,
                                                           a.frames,
                                                           a.solver_info.ptr,
                                                           a.error.ptr);
            check(cudaGetLastError());

            // Reconstruct the residual in the original temporal basis, H V_keep V_keep^H.
            // Local eigenvectors differ between tiles; averaging PCA score columns
            // directly would mix unrelated bases and arbitrary eigenvector phases.
            check(cublasCgemm(a.blas.value,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              n,
                              &one,
                              a.tile.ptr,
                              m,
                              a.covariance.ptr,
                              n,
                              &zero,
                              a.scores.ptr,
                              m));
            check(cublasCgemm(a.blas.value,
                              CUBLAS_OP_N,
                              CUBLAS_OP_C,
                              m,
                              n,
                              n,
                              &one,
                              a.scores.ptr,
                              m,
                              a.covariance.ptr,
                              n,
                              &zero,
                              a.tile.ptr,
                              m));
            accumulate_tile<<<grid_size(tile_count), 256, 0, a.stream>>>(a.tile.ptr,
                                                                         output,
                                                                         a.coverage.ptr,
                                                                         tile_count,
                                                                         a.width,
                                                                         a.height,
                                                                         a.width_x,
                                                                         a.tile_pixels,
                                                                         x,
                                                                         y);
            check(cudaGetLastError());
        }

    average_overlaps<<<grid_size(count), 256, 0, a.stream>>>(output, a.coverage.ptr, a.pixels, count);
    check(cudaGetLastError());
    // Check all eigensolves once per temporal block, avoiding a host round trip
    // for every spatial tile. Failed decompositions must not produce a displayed map.
    int error = 0;
    check(cudaMemcpyAsync(&error, a.error.ptr, sizeof(int), cudaMemcpyDeviceToHost, a.stream));
    check(cudaStreamSynchronize(a.stream));
    if (error)
        throw std::runtime_error("Local PCA eigendecomposition failed: " + std::to_string(error));
}
} // namespace holovibes::compute
