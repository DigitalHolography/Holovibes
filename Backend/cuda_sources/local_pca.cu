#include "local_pca.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cusolverDn.h>

namespace
{
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("PCA CUDA error: ") + cudaGetErrorString(status));
}
void check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("PCA cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
void check(cusolverStatus_t status, const char* operation = "operation")
{
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("Local PCA cuSOLVER ") + operation +
                                 " error: " + std::to_string(static_cast<int>(status)));
}

template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;
    ~DeviceBuffer() { cudaFree(ptr); }
    void allocate(size_t count)
    {
        if (count)
            check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
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

struct SolverParams
{
    cusolverDnParams_t value = nullptr;
    SolverParams() { check(cusolverDnCreateParams(&value)); }
    ~SolverParams() { cusolverDnDestroyParams(value); }
};

unsigned grid_size(size_t count) { return static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535)); }

// Gather a contiguous batch of spatial tiles. Modulo indexing implements periodic
// spatial borders, including corners and tiles larger than the image itself.
__global__ void gather_tile(const cuComplex* input,
                            cuComplex* tile,
                            size_t count,
                            unsigned width,
                            unsigned height,
                            unsigned width_x,
                            size_t tile_pixels,
                            size_t tile_count,
                            size_t first_tile,
                            size_t tiles_x,
                            unsigned stride_x,
                            unsigned stride_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t tile_id = first_tile + i / tile_count;
        const size_t sample = i % tile_count;
        const size_t local = sample % tile_pixels;
        const size_t origin_x = (tile_id % tiles_x) * stride_x;
        const size_t origin_y = (tile_id / tiles_x) * stride_y;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        tile[i] = input[(sample / tile_pixels) * pixels + static_cast<size_t>(y) * width + x];
    }
}

// cuSOLVER orders the eigenvalues increasingly. Compute the Shannon effective
// rank of the Gram matrix, e = exp(-sum(p * log(p))), p = lambda/sum(lambda),
// and choose r = min(frames, ceil(rank_factor * e)). Scale BEFORE rounding so
// small factor adjustments can move the cutoff even when e is not an integer.
// Then zero the LAST r eigenvectors to remove the strongest components. A zero
// spectrum has rank zero; negative roundoff eigenvalues carry no probability.
__global__ void remove_effective_rank(cuComplex* vectors,
                                      const float* eigenvalues,
                                      unsigned frames,
                                      const int* solver_info,
                                      int* error,
                                      float rank_factor)
{
    // One CUDA block per independent eigensystem.
    vectors += static_cast<size_t>(blockIdx.x) * frames * frames;
    eigenvalues += static_cast<size_t>(blockIdx.x) * frames;
    solver_info += blockIdx.x;
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
        if (total > 0. && rank_factor > 0.f)
        {
            double entropy = 0.;
            for (unsigned i = 0; i < frames; ++i)
            {
                const double p = fmax(0., static_cast<double>(eigenvalues[i])) / total;
                if (p > 0.)
                    entropy -= p * log(p);
            }
            // Avoid rounding a numerically perturbed integer rank up by one.
            const double scaled_rank = static_cast<double>(rank_factor) * exp(entropy);
            removed = static_cast<unsigned>(fmax(1., fmin(static_cast<double>(frames), ceil(scaled_rank - 1e-6))));
        }
    }
    __syncthreads();

    const size_t first_removed = static_cast<size_t>(frames - removed) * frames;
    const size_t size = static_cast<size_t>(frames) * frames;
    for (size_t i = first_removed + threadIdx.x; i < size; i += blockDim.x)
        vectors[i] = make_cuComplex(0.f, 0.f);
}

// The number of spatial contributions depends only on geometry. Count whole
// wraps, then count lattice origins in the remaining periodic interval.
__device__ unsigned axis_coverage(size_t position, size_t length, size_t side, size_t stride)
{
    const size_t origins = (length + stride - 1) / stride;
    size_t contributions = (side / length) * origins;
    const size_t remainder = side % length;
    if (remainder)
    {
        const size_t right = position / stride + 1;
        if (position + 1 >= remainder)
            contributions += right - (position + 1 - remainder + stride - 1) / stride;
        else
            contributions += right + origins - (length + position + 1 - remainder + stride - 1) / stride;
    }
    return static_cast<unsigned>(contributions);
}

__global__ void initialize_coverage(unsigned* coverage,
                                    size_t pixels,
                                    unsigned width,
                                    unsigned height,
                                    unsigned width_x,
                                    unsigned width_y,
                                    unsigned stride_x,
                                    unsigned stride_y)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pixels;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        coverage[i] =
            axis_coverage(i % width, width, width_x, stride_x) * axis_coverage(i / width, height, width_y, stride_y);
}

// Exact partitions have one writer per output sample. Overlapping/wrapped tiles
// can race both within and across tiles and require atomic accumulation.
template <bool direct>
__global__ void accumulate_tile(const cuComplex* tile,
                                cuComplex* output,
                                size_t count,
                                unsigned width,
                                unsigned height,
                                unsigned width_x,
                                size_t tile_pixels,
                                size_t tile_count,
                                size_t first_tile,
                                size_t tiles_x,
                                unsigned stride_x,
                                unsigned stride_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t tile_id = first_tile + i / tile_count;
        const size_t sample = i % tile_count;
        const size_t local = sample % tile_pixels;
        const size_t origin_x = (tile_id % tiles_x) * stride_x;
        const size_t origin_y = (tile_id / tiles_x) * stride_y;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        const size_t pixel = static_cast<size_t>(y) * width + x;
        const size_t target = (sample / tile_pixels) * pixels + pixel;
        if constexpr (direct)
            output[target] = tile[i];
        else
        {
            atomicAdd(&output[target].x, tile[i].x);
            atomicAdd(&output[target].y, tile[i].y);
        }
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

__global__ void sum_covariance_parts(cuComplex* covariance, const cuComplex* parts, size_t count, int part_count)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        double real = 0., imag = 0.;
        for (int part = 0; part < part_count; ++part)
        {
            const cuComplex value = parts[part * count + i];
            real += value.x;
            imag += value.y;
        }
        covariance[i] = make_cuComplex(static_cast<float>(real), static_cast<float>(imag));
    }
}

void check_pca_eigendecomposition(const int* device_error, cudaStream_t stream)
{
    int error = 0;
    check(cudaMemcpyAsync(&error, device_error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (error)
        throw std::runtime_error("PCA eigendecomposition failed: " + std::to_string(error));
}
} // namespace

namespace holovibes::compute
{
void pca_gram_matrix(cublasHandle_t blas, const cuComplex* samples, int pixels, int frames, cuComplex* covariance)
{
    const cuComplex one{1.f, 0.f}, zero{0.f, 0.f};
    check(cublasCgemm(blas,
                      CUBLAS_OP_C,
                      CUBLAS_OP_N,
                      frames,
                      frames,
                      pixels,
                      &one,
                      samples,
                      pixels,
                      samples,
                      pixels,
                      &zero,
                      covariance,
                      frames));
}

void remove_pca_effective_rank(cuComplex* vectors,
                               const float* eigenvalues,
                               unsigned frames,
                               int* solver_info,
                               float rank_factor,
                               cudaStream_t stream)
{
    // With one eigensystem its solver status can also hold the spectrum check result.
    remove_effective_rank<<<1, 256, 0, stream>>>(vectors, eigenvalues, frames, solver_info, solver_info, rank_factor);
    check(cudaGetLastError());
    check_pca_eigendecomposition(solver_info, stream);
}

struct LocalPca::Impl
{
    unsigned width, height, frames, width_x, width_y, stride_x, stride_y;
    size_t pixels, tile_pixels, tiles_x, tiles;
    unsigned batch_capacity;
    int covariance_rows, covariance_part_count;
    bool direct_output;
    cudaStream_t stream;
    BlasHandle blas;
    SolverHandle solver;
    SolverParams solver_params;
    DeviceBuffer<cuComplex> tile, scores, covariance, covariance_parts;
    DeviceBuffer<unsigned char> workspace;
    std::vector<unsigned char> host_workspace;
    DeviceBuffer<float> eigenvalues;
    DeviceBuffer<unsigned> coverage;
    DeviceBuffer<int> solver_info, error;
    int legacy_workspace_size = 0;
    size_t workspace_bytes = 0, host_workspace_bytes = 0;

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

        tiles_x = (static_cast<size_t>(width) + stride_x - 1) / stride_x;
        tiles = tiles_x * ((static_cast<size_t>(height) + stride_y - 1) / stride_y);
        direct_output = stride_x == width_x && stride_y == width_y && width % width_x == 0 && height % width_y == 0;
        const size_t matrix_size = static_cast<size_t>(frames) * frames;
        // Shorter dot products keep float accumulation errors small in a weak spectrum.
        // Bound the number of partial matrices and their size relative to the tile data.
        constexpr size_t max_covariance_parts = 8;
        covariance_rows = static_cast<int>(std::max({size_t{64}, size_t{frames},
                                                     (tile_pixels + max_covariance_parts - 1) / max_covariance_parts}));
        covariance_part_count = static_cast<int>((tile_pixels + covariance_rows - 1) / covariance_rows);
        const size_t partial_matrix_size = covariance_part_count > 1 ? matrix_size * covariance_part_count : 0;
        // Bound tile storage independently of image size and spatial overlap.
        // One unusually large tile still uses the original single-tile solver.
        constexpr size_t scratch_budget = 64 * 1024 * 1024;
        const size_t bytes_per_tile =
            (2 * tile_pixels * frames + matrix_size + partial_matrix_size) * sizeof(cuComplex) + frames * sizeof(float) +
            sizeof(int);
        const size_t memory_limit = std::max<size_t>(1, scratch_budget / bytes_per_tile);
        // XsyevBatched requires n * lda * batchSize <= INT32_MAX.
        const size_t solver_limit = std::max<size_t>(1, std::numeric_limits<int>::max() / matrix_size);
        batch_capacity = static_cast<unsigned>(std::min({tiles, size_t{64}, memory_limit, solver_limit}));
        tile.allocate(tile_pixels * frames * batch_capacity);
        scores.allocate(tile_pixels * frames * batch_capacity);
        covariance.allocate(matrix_size * batch_capacity);
        if (batch_capacity > 1)
            covariance_parts.allocate(partial_matrix_size * batch_capacity);
        eigenvalues.allocate(static_cast<size_t>(frames) * batch_capacity);
        solver_info.allocate(batch_capacity);
        error.allocate(1);
        if (batch_capacity == 1)
        {
            check(cusolverDnCheevd_bufferSize(solver.value,
                                              CUSOLVER_EIG_MODE_VECTOR,
                                              CUBLAS_FILL_MODE_LOWER,
                                              frames,
                                              covariance.ptr,
                                              frames,
                                              eigenvalues.ptr,
                                              &legacy_workspace_size));
            workspace_bytes = static_cast<size_t>(legacy_workspace_size) * sizeof(cuComplex);
        }
        else
        {
            // CUDA 13.0 rejects algorithm selection for this routine (NVIDIA
            // issue 5539844, fixed in 13.1). Its default remains usable.
            const auto options_status =
                cusolverDnSetAdvOptions(solver_params.value, CUSOLVERDN_SYEVBATCHED, CUSOLVER_ALG_1);
            if (options_status != CUSOLVER_STATUS_INVALID_VALUE && options_status != CUSOLVER_STATUS_NOT_SUPPORTED)
                check(options_status, "batched options");
            const auto query_workspace = [&](size_t batch)
            {
                size_t device_bytes = 0, host_bytes = 0;
                check(cusolverDnXsyevBatched_bufferSize(solver.value,
                                                        solver_params.value,
                                                        CUSOLVER_EIG_MODE_VECTOR,
                                                        CUBLAS_FILL_MODE_LOWER,
                                                        frames,
                                                        CUDA_C_32F,
                                                        covariance.ptr,
                                                        frames,
                                                        CUDA_R_32F,
                                                        eigenvalues.ptr,
                                                        CUDA_C_32F,
                                                        &device_bytes,
                                                        &host_bytes,
                                                        batch),
                      "batched workspace query");
                workspace_bytes = std::max(workspace_bytes, device_bytes);
                host_workspace_bytes = std::max(host_workspace_bytes, host_bytes);
            };
            query_workspace(batch_capacity);
            if (tiles % batch_capacity)
                query_workspace(tiles % batch_capacity);
        }
        workspace.allocate(workspace_bytes);
        host_workspace.resize(host_workspace_bytes);
        if (!direct_output)
        {
            coverage.allocate(pixels);
            initialize_coverage<<<grid_size(pixels), 256, 0, stream>>>(coverage.ptr,
                                                                       pixels,
                                                                       width,
                                                                       height,
                                                                       width_x,
                                                                       width_y,
                                                                       stride_x,
                                                                       stride_y);
            check(cudaGetLastError());
        }
    }

    void compute_covariance(int batch)
    {
        const int m = static_cast<int>(tile_pixels), n = static_cast<int>(frames);
        if (batch_capacity == 1)
        {
            pca_gram_matrix(blas.value, tile.ptr, m, n, covariance.ptr);
            return;
        }

        const cuComplex one{1.f, 0.f}, zero{0.f, 0.f};
        const long long tile_stride = static_cast<long long>(m) * n;
        const long long covariance_stride = static_cast<long long>(n) * n;
        const size_t count = static_cast<size_t>(covariance_stride) * batch;
        for (int part = 0; part < covariance_part_count; ++part)
        {
            const int first_row = part * covariance_rows;
            const auto* samples = tile.ptr + first_row;
            auto* result = covariance_parts.ptr ? covariance_parts.ptr + part * count : covariance.ptr;
            check(cublasCgemmStridedBatched(blas.value,
                                            CUBLAS_OP_C,
                                            CUBLAS_OP_N,
                                            n,
                                            n,
                                            std::min(covariance_rows, m - first_row),
                                            &one,
                                            samples,
                                            m,
                                            tile_stride,
                                            samples,
                                            m,
                                            tile_stride,
                                            &zero,
                                            result,
                                            n,
                                            covariance_stride,
                                            batch));
        }
        if (covariance_parts.ptr)
        {
            sum_covariance_parts<<<grid_size(count), 256, 0, stream>>>(covariance.ptr,
                                                                      covariance_parts.ptr,
                                                                      count,
                                                                      covariance_part_count);
            check(cudaGetLastError());
        }
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

void LocalPca::process(const cuComplex* input, cuComplex* output, float rank_factor)
{
    if (!input || !output || input == output)
        throw std::invalid_argument("Local PCA requires distinct, nonnull input and output buffers");
    if (!std::isfinite(rank_factor) || rank_factor < 0.f)
        throw std::invalid_argument("Local PCA rank factor must be finite and nonnegative");
    auto& a = *impl_;
    const cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    const size_t tile_count = a.tile_pixels * a.frames, count = a.pixels * a.frames;
    // At factor zero, every local projector is the identity and overlap averaging
    // gives the original cube. Copy it directly to avoid needless eigensolves.
    if (rank_factor == 0.f)
    {
        check(cudaMemcpyAsync(output, input, count * sizeof(cuComplex), cudaMemcpyDeviceToDevice, a.stream));
        check(cudaStreamSynchronize(a.stream));
        return;
    }
    const int m = static_cast<int>(a.tile_pixels), n = static_cast<int>(a.frames);
    if (!a.direct_output)
        check(cudaMemsetAsync(output, 0, count * sizeof(cuComplex), a.stream));
    check(cudaMemsetAsync(a.error.ptr, 0, sizeof(int), a.stream));

    const long long tile_stride = static_cast<long long>(tile_count);
    const long long covariance_stride = static_cast<long long>(n) * n;
    for (size_t first_tile = 0; first_tile < a.tiles; first_tile += a.batch_capacity)
    {
        const int batch = static_cast<int>(std::min<size_t>(a.batch_capacity, a.tiles - first_tile));
        const size_t batch_count = tile_count * batch;
        gather_tile<<<grid_size(batch_count), 256, 0, a.stream>>>(input,
                                                                  a.tile.ptr,
                                                                  batch_count,
                                                                  a.width,
                                                                  a.height,
                                                                  a.width_x,
                                                                  a.tile_pixels,
                                                                  tile_count,
                                                                  first_tile,
                                                                  a.tiles_x,
                                                                  a.stride_x,
                                                                  a.stride_y);
        check(cudaGetLastError());

        // Match classic PCA's uncentered complex covariance: C = H^H H.
        a.compute_covariance(batch);
        if (a.batch_capacity == 1)
            check(cusolverDnCheevd(a.solver.value,
                                   CUSOLVER_EIG_MODE_VECTOR,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   a.covariance.ptr,
                                   n,
                                   a.eigenvalues.ptr,
                                   reinterpret_cast<cuComplex*>(a.workspace.ptr),
                                   a.legacy_workspace_size,
                                   a.solver_info.ptr));
        else
            check(cusolverDnXsyevBatched(a.solver.value,
                                         a.solver_params.value,
                                         CUSOLVER_EIG_MODE_VECTOR,
                                         CUBLAS_FILL_MODE_LOWER,
                                         n,
                                         CUDA_C_32F,
                                         a.covariance.ptr,
                                         n,
                                         CUDA_R_32F,
                                         a.eigenvalues.ptr,
                                         CUDA_C_32F,
                                         a.workspace.ptr,
                                         a.workspace_bytes,
                                         a.host_workspace.empty() ? nullptr : a.host_workspace.data(),
                                         a.host_workspace_bytes,
                                         a.solver_info.ptr,
                                         batch),
                  "batched eigensolve");
        remove_effective_rank<<<batch, 256, 0, a.stream>>>(a.covariance.ptr,
                                                           a.eigenvalues.ptr,
                                                           a.frames,
                                                           a.solver_info.ptr,
                                                           a.error.ptr,
                                                           rank_factor);
        check(cudaGetLastError());

        // Reconstruct the residual in the original temporal basis, H V_keep V_keep^H.
        // Local eigenvectors differ between tiles; averaging PCA score columns
        // directly would mix unrelated bases and arbitrary eigenvector phases.
        check(cublasCgemmStridedBatched(a.blas.value,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        m,
                                        n,
                                        n,
                                        &one,
                                        a.tile.ptr,
                                        m,
                                        tile_stride,
                                        a.covariance.ptr,
                                        n,
                                        covariance_stride,
                                        &zero,
                                        a.scores.ptr,
                                        m,
                                        tile_stride,
                                        batch));
        check(cublasCgemmStridedBatched(a.blas.value,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_C,
                                        m,
                                        n,
                                        n,
                                        &one,
                                        a.scores.ptr,
                                        m,
                                        tile_stride,
                                        a.covariance.ptr,
                                        n,
                                        covariance_stride,
                                        &zero,
                                        a.tile.ptr,
                                        m,
                                        tile_stride,
                                        batch));
        if (a.direct_output)
            accumulate_tile<true><<<grid_size(batch_count), 256, 0, a.stream>>>(a.tile.ptr,
                                                                                output,
                                                                                batch_count,
                                                                                a.width,
                                                                                a.height,
                                                                                a.width_x,
                                                                                a.tile_pixels,
                                                                                tile_count,
                                                                                first_tile,
                                                                                a.tiles_x,
                                                                                a.stride_x,
                                                                                a.stride_y);
        else
            accumulate_tile<false><<<grid_size(batch_count), 256, 0, a.stream>>>(a.tile.ptr,
                                                                                 output,
                                                                                 batch_count,
                                                                                 a.width,
                                                                                 a.height,
                                                                                 a.width_x,
                                                                                 a.tile_pixels,
                                                                                 tile_count,
                                                                                 first_tile,
                                                                                 a.tiles_x,
                                                                                 a.stride_x,
                                                                                 a.stride_y);
        check(cudaGetLastError());
    }

    if (!a.direct_output)
    {
        average_overlaps<<<grid_size(count), 256, 0, a.stream>>>(output, a.coverage.ptr, a.pixels, count);
        check(cudaGetLastError());
    }
    // Check all eigensolves once per temporal block, avoiding a host round trip
    // for every spatial tile. Failed decompositions must not produce a displayed map.
    check_pca_eigendecomposition(a.error.ptr, a.stream);
}
} // namespace holovibes::compute
