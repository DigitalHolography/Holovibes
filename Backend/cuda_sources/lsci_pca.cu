#include "lsci_pca.hh"
#include "lsci.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <opencv2/core.hpp>

namespace
{
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Offline PCA-LSCI CUDA error: ") + cudaGetErrorString(status));
}
void check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Offline PCA-LSCI cuBLAS error: " + std::to_string(static_cast<int>(status)));
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
    BlasHandle()
    {
        check(cublasCreate(&value));
    }
    ~BlasHandle() { cublasDestroy(value); }
};

__global__ void fill_ones(float* values, size_t count)
{
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < count)
        values[p] = 1.f;
}

template <typename T>
__global__ void store_frame(const T* input, float* output, size_t count, int* invalid)
{
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < count)
    {
        const float value = static_cast<float>(input[p]);
        if (!isfinite(value) || value < 0.f)
            atomicExch(invalid, 1);
        output[p] = value;
    }
}

__global__ void center_frames(float* stack, const float* means, size_t pixels, size_t count)
{
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < count)
        stack[p] -= means[p / pixels];
}

__global__ void extract_patch(const float* stack, float* patch, size_t pixels, unsigned width,
                              unsigned x, unsigned y, unsigned patch_width, size_t patch_pixels,
                              size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const size_t p = index % patch_pixels;
        const size_t image_pixel = (y + p / patch_width) * width + x + p % patch_width;
        patch[index] = stack[(index / patch_pixels) * pixels + image_pixel];
    }
}

__global__ void place_patch_moments(const float2* patch, float2* image, unsigned width,
                                    unsigned x, unsigned y, unsigned patch_width, size_t patch_pixels)
{
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < patch_pixels)
        image[(y + p / patch_width) * width + x + p % patch_width] = patch[p];
}

__global__ void static_moments(float2* output,
                               const float* scores,
                               const float2* coefficients,
                               unsigned pixels,
                               unsigned components,
                               double mean_sum,
                               double mean_squared_sum)
{
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels)
        return;
    double sum = mean_sum;
    double squared = mean_squared_sum;
    for (unsigned c = 0; c < components; ++c)
    {
        const double score = scores[static_cast<size_t>(c) * pixels + p];
        const float2 coefficient = coefficients[c];
        sum += score * coefficient.x;
        squared += score * score + 2. * score * coefficient.y;
    }
    output[p] = make_float2(static_cast<float>(sum), static_cast<float>(squared));
}
}

namespace holovibes::compute
{
struct LsciPca::Impl
{
    unsigned width, height, frames, window, patch_size, loaded = 0;
    size_t pixels;
    BlasHandle blas;
    DeviceBuffer<float> stack, ones, means, gram, basis, scores, original_map, static_map;
    DeviceBuffer<float2> original_moments, reconstructed_moments, coefficients;
    DeviceBuffer<float> patch_stack;
    DeviceBuffer<float2> patch_moments;
    DeviceBuffer<unsigned char> input;
    DeviceBuffer<int> invalid;
    static constexpr unsigned tile_size = 65536;

    Impl(unsigned w, unsigned h, unsigned n, unsigned s, unsigned patch)
        : width(w), height(h), frames(n), window(s), patch_size(patch), pixels(static_cast<size_t>(w) * h)
    {
        if (!w || !h || pixels > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            n < 2 || n > 512 || !s || s > 63 || s % 2 == 0)
            throw std::invalid_argument("PCA-LSCI requires 2..512 frames, an odd 1..63 spatial window and valid dimensions");
        if (patch == 1 || patch > 512)
            throw std::invalid_argument("PCA-LSCI patch size must be 0 (global) or 2..512 pixels");
        const size_t patch_pixels = patch ? static_cast<size_t>(std::min(w, patch)) * std::min(h, patch) : 0;
        size_t available, total;
        check(cudaMemGetInfo(&available, &total));
        const size_t required = sizeof(float) * (pixels * (n + 8ull) +
                                                static_cast<size_t>(tile_size) * n + 2ull * n * n + 3ull * n +
                                                patch_pixels * (n + 2ull));
        if (required + 16ull * 1024 * 1024 > available)
            throw std::runtime_error("Not enough GPU memory for PCA-LSCI (approximately " +
                                     std::to_string(required / (1024 * 1024)) +
                                     " MiB required). Reduce --lsci_pca_frames or image dimensions.");
        check(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH));
        stack.allocate(pixels * n);
        ones.allocate(pixels);
        means.allocate(n);
        gram.allocate(static_cast<size_t>(n) * n);
        basis.allocate(static_cast<size_t>(n) * n);
        scores.allocate(static_cast<size_t>(tile_size) * n);
        original_map.allocate(pixels);
        static_map.allocate(pixels);
        original_moments.allocate(pixels);
        reconstructed_moments.allocate(pixels);
        coefficients.allocate(n);
        if (patch)
        {
            patch_stack.allocate(patch_pixels * n);
            patch_moments.allocate(patch_pixels);
        }
        input.allocate(pixels * sizeof(float));
        invalid.allocate(1);
        check(cudaMemset(invalid.ptr, 0, sizeof(int)));
        fill_ones<<<static_cast<unsigned>((pixels + 255) / 256), 256>>>(ones.ptr, pixels);
        check(cudaGetLastError());
    }

    void reconstruct(float* data, unsigned domain_pixels, float2* output,
                      std::vector<double>& eigenvalues, unsigned& components);
};

LsciPca::LsciPca(unsigned width, unsigned height, unsigned frames, unsigned spatial_window, unsigned pca_patch_size)
    : impl_(std::make_unique<Impl>(width, height, frames, spatial_window, pca_patch_size))
{
}
LsciPca::~LsciPca() = default;

void LsciPca::add_frame(const void* host_frame, camera::PixelDepth depth)
{
    auto& a = *impl_;
    if (a.loaded >= a.frames)
        throw std::logic_error("PCA-LSCI block is already full");
    if (depth != camera::PixelDepth::Bits8 && depth != camera::PixelDepth::Bits16 && depth != camera::PixelDepth::Bits32)
        throw std::invalid_argument("PCA-LSCI supports 8-bit, 16-bit and float32 intensities");
    check(cudaMemcpy(a.input.ptr, host_frame, a.pixels * depth, cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>((a.pixels + 255) / 256);
    float* target = a.stack.ptr + a.loaded * a.pixels;
    if (depth == camera::PixelDepth::Bits8)
        store_frame<<<blocks, 256>>>(a.input.ptr, target, a.pixels, a.invalid.ptr);
    else if (depth == camera::PixelDepth::Bits16)
        store_frame<<<blocks, 256>>>(reinterpret_cast<unsigned short*>(a.input.ptr), target, a.pixels, a.invalid.ptr);
    else
        store_frame<<<blocks, 256>>>(reinterpret_cast<float*>(a.input.ptr), target, a.pixels, a.invalid.ptr);
    check(cudaGetLastError());
    lsci_accumulate(a.original_moments.ptr, a.input.ptr, a.pixels, 1, depth, a.loaded == 0, nullptr);
    check(cudaGetLastError());
    ++a.loaded;
}

void LsciPca::Impl::reconstruct(float* data, unsigned domain_pixels, float2* output,
                               std::vector<double>& spectrum, unsigned& components)
{
    const int m = static_cast<int>(domain_pixels), n = static_cast<int>(frames);
    const float one = 1.f, zero = 0.f, scale = 1.f / m;
    // Preserve the existing spatial centering, eigenvalue rule and moment
    // reconstruction, applied either to the whole image or one local patch.
    check(cublasSgemv(blas.value, CUBLAS_OP_T, m, n, &scale, data, m, ones.ptr, 1,
                      &zero, means.ptr, 1));
    std::vector<float> host_means(n);
    check(cudaMemcpy(host_means.data(), means.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    const size_t count = static_cast<size_t>(domain_pixels) * n;
    center_frames<<<static_cast<unsigned>((count + 255) / 256), 256>>>(data, means.ptr, domain_pixels, count);
    check(cudaGetLastError());
    // The N x N Gram matrix has the same nonzero eigenvalues as Phi Phi^T.
    check(cublasSgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &one,
                      data, m, data, m, &zero, gram.ptr, n));
    std::vector<float> host_gram(static_cast<size_t>(n) * n);
    check(cudaMemcpy(host_gram.data(), gram.ptr, host_gram.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat covariance(n, n, CV_64F);
    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column)
            covariance.at<double>(row, column) = .5 * (host_gram[column*n+row] + host_gram[row*n+column]);
    if (!cv::checkRange(covariance))
        throw std::runtime_error("PCA covariance overflowed; rescale float32 input intensities");
    cv::Mat eigenvalues, eigenvectors;
    if (!cv::eigen(covariance, eigenvalues, eigenvectors))
        throw std::runtime_error("PCA eigendecomposition did not converge");
    spectrum.clear();
    spectrum.reserve(n);
    double total = 0.;
    for (int i = 0; i < n; ++i)
    {
        const double value = std::max(0., eigenvalues.at<double>(i));
        spectrum.push_back(value);
        total += value;
    }
    const double threshold = total / n;
    std::vector<float> basis;
    std::vector<float2> coefficients;
    // cv::eigen returns eigenvectors as rows; cuBLAS consumes columns.
    if (total > 0.)
        for (int i = 0; i < n; ++i)
            if (spectrum[i] >= threshold)
            {
                double sum = 0., mean_dot = 0.;
                for (int t = 0; t < n; ++t)
                {
                    const float v = static_cast<float>(eigenvectors.at<double>(i, t));
                    basis.push_back(v);
                    sum += v;
                    mean_dot += v * host_means[t];
                }
                coefficients.push_back(make_float2(static_cast<float>(sum), static_cast<float>(mean_dot)));
            }
    components = static_cast<unsigned>(coefficients.size());
    if (!basis.empty())
    {
        check(cudaMemcpy(this->basis.ptr, basis.data(), basis.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(this->coefficients.ptr, coefficients.data(), coefficients.size() * sizeof(float2), cudaMemcpyHostToDevice));
    }
    const double mean_sum = std::accumulate(host_means.begin(), host_means.end(), 0.);
    double mean_squared_sum = 0.;
    for (float mean : host_means)
        mean_squared_sum += static_cast<double>(mean) * mean;
    for (size_t first = 0; first < domain_pixels; first += tile_size)
    {
        const int rows = static_cast<int>(std::min<size_t>(tile_size, domain_pixels - first));
        const int k = static_cast<int>(components);
        if (k)
            check(cublasSgemm(blas.value, CUBLAS_OP_N, CUBLAS_OP_N, rows, k, n, &one,
                              data + first, m, this->basis.ptr, n, &zero, scores.ptr, rows));
        // For S = mu + (Phi V_s)V_s^T, compute temporal moments exactly from
        // orthonormal scores; no second full M x N reconstruction is allocated.
        static_moments<<<(rows + 255) / 256, 256>>>(output + first, scores.ptr,
                                                    this->coefficients.ptr, rows, k, mean_sum, mean_squared_sum);
        check(cudaGetLastError());
    }
}

LsciPcaResult LsciPca::process()
{
    auto& a = *impl_;
    if (a.loaded != a.frames)
        throw std::logic_error("PCA-LSCI needs a complete frame block");
    int invalid;
    check(cudaMemcpy(&invalid, a.invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid)
        throw std::invalid_argument("PCA-LSCI requires finite, nonnegative input intensities");

    LsciPcaResult result;
    if (!a.patch_size)
        a.reconstruct(a.stack.ptr, static_cast<unsigned>(a.pixels), a.reconstructed_moments.ptr,
                       result.eigenvalues, result.static_components);
    else
    {
        const size_t columns = 1 + (a.width - 1) / a.patch_size;
        const size_t rows = 1 + (a.height - 1) / a.patch_size;
        result.patches.reserve(columns * rows);
        for (unsigned y = 0; y < a.height; y += a.patch_size)
            for (unsigned x = 0; x < a.width; x += a.patch_size)
            {
                const unsigned w = std::min(a.patch_size, a.width - x);
                const unsigned h = std::min(a.patch_size, a.height - y);
                const unsigned pixels = w * h;
                const size_t count = static_cast<size_t>(pixels) * a.frames;
                extract_patch<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
                    a.stack.ptr, a.patch_stack.ptr, a.pixels, a.width, x, y, w, pixels, count);
                check(cudaGetLastError());
                auto& patch = result.patches.emplace_back(LsciPcaPatchResult{x, y, w, h});
                a.reconstruct(a.patch_stack.ptr, pixels, a.patch_moments.ptr,
                               patch.eigenvalues, patch.static_components);
                place_patch_moments<<<(pixels + 255) / 256, 256>>>(
                    a.patch_moments.ptr, a.reconstructed_moments.ptr, a.width, x, y, w, pixels);
                check(cudaGetLastError());
            }
    }
    // Assemble all local moments before contrast: the LSCI neighborhood may
    // cross PCA patch boundaries and is clipped only at the full-image edges.
    lsci_contrast(a.original_map.ptr, a.original_moments.ptr, a.width, a.height, a.frames, nullptr, a.window);
    lsci_contrast(a.static_map.ptr, a.reconstructed_moments.ptr, a.width, a.height, a.frames, nullptr, a.window);
    check(cudaGetLastError());
    result.original.resize(a.pixels);
    result.static_component.resize(a.pixels);
    result.dynamic.resize(a.pixels);
    check(cudaMemcpy(result.original.data(), a.original_map.ptr, a.pixels * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result.static_component.data(), a.static_map.ptr, a.pixels * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t p = 0; p < a.pixels; ++p)
    {
        // Map subtraction, not LSCI of the residual and not squared-contrast subtraction.
        result.dynamic[p] = result.original[p] - result.static_component[p];
        if (!std::isfinite(result.dynamic[p]))
            throw std::runtime_error("PCA-LSCI produced a nonfinite contrast map");
    }
    a.loaded = 0;
    return result;
}
}
