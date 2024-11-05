#include "analysis.hh"
#include "icompute.hh"

#include "convolution.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "contrast_correction.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"
#include "holovibes.hh"
#include "otsu.cuh"
#include "cublas_handle.hh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

using holovibes::cuda_tools::CufftHandle;

namespace holovibes::compute
{

namespace // petit flex
{
std::vector<float> load_convolution_matrix()
{
    // There is no file None.txt for convolution
    std::vector<float> convo_matrix = {};
    const std::string& file = "gaussian_128_128_1.txt";
    auto& holo = holovibes::Holovibes::instance();

    try
    {
        auto path_file = GET_EXE_DIR / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file; //"convolution_kernels" / file;
        std::string path = path_file.string();

        std::vector<float> matrix;
        uint matrix_width = 0;
        uint matrix_height = 0;
        uint matrix_z = 1;

        // Doing this the C way because it's faster
        FILE* c_file;
        fopen_s(&c_file, path.c_str(), "r");

        if (c_file == nullptr)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid file path");
        }

        // Read kernel dimensions
        if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid kernel dimensions");
        }

        size_t matrix_size = matrix_width * matrix_height * matrix_z;
        matrix.resize(matrix_size);

        // Read kernel values
        for (size_t i = 0; i < matrix_size; ++i)
        {
            if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
            {
                fclose(c_file);
                throw std::runtime_error("Missing values");
            }
        }

        fclose(c_file);

        // Reshape the vector as a (nx,ny) rectangle, keeping z depth
        const uint output_width = holo.get_gpu_output_queue()->get_fd().width;
        const uint output_height = holo.get_gpu_output_queue()->get_fd().height;
        const uint size = output_width * output_height;

        // The convo matrix is centered and padded with 0 since the kernel is
        // usally smaller than the output Example: kernel size is (2, 2) and
        // output size is (4, 4) The kernel is represented by 'x' and
        //  | 0 | 0 | 0 | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | 0 | 0 | 0 |
        const uint first_col = (output_width / 2) - (matrix_width / 2);
        const uint last_col = (output_width / 2) + (matrix_width / 2);
        const uint first_row = (output_height / 2) - (matrix_height / 2);
        const uint last_row = (output_height / 2) + (matrix_height / 2);

        convo_matrix.resize(size, 0.0f);

        uint kernel_indice = 0;
        for (uint i = first_row; i < last_row; i++)
        {
            for (uint j = first_col; j < last_col; j++)
            {
                (convo_matrix)[i * output_width + j] = matrix[kernel_indice];
                kernel_indice++;
            }
        }
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
        return {};
    };
    return convo_matrix;
}

} // namespace

void Analysis::init()
{
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // No need for memset here since it will be completely overwritten by
    // cuComplex values
    buffers_.gpu_convolution_buffer.resize(frame_res);

    // No need for memset here since it will be memset in the actual convolution
    cuComplex_buffer_.resize(frame_res);

    gaussian_kernel_ = load_convolution_matrix();

    gpu_kernel_buffer_.resize(frame_res);
    cudaXMemsetAsync(gpu_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(gpu_kernel_buffer_.get(),
                                   sizeof(cuComplex),
                                   gaussian_kernel_.data(),
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyHostToDevice,
                                   stream_));

    constexpr uint batch_size = 1; // since only one frame.
    // We compute the FFT of the kernel, once, here, instead of every time the
    // convolution subprocess is called
    shift_corners(gpu_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_, gpu_kernel_buffer_.get(), gpu_kernel_buffer_.get(), CUFFT_FORWARD));
    number_image_mean_ = 0;

    // 100 = batch_moment for analysis
    buffer_m0_ff_img_ = new float*[number_hardcode_];
    for (int i = 0; i < number_hardcode_; i++)
    {
        buffer_m0_ff_img_[i] = new float[buffers_.gpu_postprocess_frame_size];
    }
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gpu_kernel_buffer_.reset(nullptr);
}

void Analysis::insert_show_artery()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>() == true)
            {
                convolution_kernel(buffers_.gpu_postprocess_frame,
                                   buffers_.gpu_convolution_buffer,
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gpu_kernel_buffer_.get(),
                                   true,
                                   stream_);
                number_image_mean_++;
                if (number_image_mean_ == 1)
                {
                    cudaXMemcpy(buffer_m0_ff_img_[0],
                                buffers_.gpu_postprocess_frame,
                                buffers_.gpu_postprocess_frame_size * sizeof(float),
                                cudaMemcpyDeviceToHost);

                    m0_ff_sum_image_ = new float[buffers_.gpu_postprocess_frame_size];
                    std::memcpy(m0_ff_sum_image_,
                                buffer_m0_ff_img_[0],
                                buffers_.gpu_postprocess_frame_size * sizeof(float));
                }
                else
                {
                    float* new_image = new float[buffers_.gpu_postprocess_frame_size];
                    cudaXMemcpy(new_image,
                                buffers_.gpu_postprocess_frame,
                                buffers_.gpu_postprocess_frame_size * sizeof(float),
                                cudaMemcpyDeviceToHost);
                    if (number_image_mean_ >= number_hardcode_)
                    {
                        for (uint i = 0; i < buffers_.gpu_postprocess_frame_size; i++)
                        {
                            m0_ff_sum_image_[i] -= buffer_m0_ff_img_[(number_image_mean_ - 1) % number_hardcode_][i];
                        }
                    }
                    std::memcpy(buffer_m0_ff_img_[(number_image_mean_ - 1) % number_hardcode_],
                                new_image,
                                buffers_.gpu_postprocess_frame_size * sizeof(float));

                    for (uint i = 0; i < buffers_.gpu_postprocess_frame_size; i++)
                    {
                        m0_ff_sum_image_[i] += new_image[i];
                    }
                    // TODO its not 100 it s batch_moment for analysis
                    if (number_image_mean_ >= number_hardcode_)
                    {
                        for (uint i = 0; i < buffers_.gpu_postprocess_frame_size; i++)
                        {
                            new_image[i] = m0_ff_sum_image_[i] / number_hardcode_;
                        }
                        cudaXMemcpy(buffers_.gpu_postprocess_frame,
                                    new_image,
                                    buffers_.gpu_postprocess_frame_size * sizeof(float),
                                    cudaMemcpyHostToDevice);
                    }
                    delete new_image;
                }
            }
        });
}

void Analysis::insert_otsu()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::OtsuEnabled>() == true)
            {

                cublasHandle_t& handle = cuda_tools::CublasHandle::instance();
                int maxI = -1;
                int minI = -1;
                cublasIsamax(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &maxI);
                cublasIsamin(handle, buffers_.gpu_postprocess_frame_size, buffers_.gpu_postprocess_frame, 1, &minI);

                float h_min, h_max;
                cudaXMemcpy(&h_min, buffers_.gpu_postprocess_frame + (minI - 1), sizeof(float), cudaMemcpyDeviceToHost);
                cudaXMemcpy(&h_max, buffers_.gpu_postprocess_frame + (maxI - 1), sizeof(float), cudaMemcpyDeviceToHost);

                normalise(buffers_.gpu_postprocess_frame, h_min, h_max, buffers_.gpu_postprocess_frame_size, stream_);

                if (setting<settings::OtsuKind>() == OtsuKind::Adaptive)
                {

                    float* d_output;
                    cudaMalloc(&d_output, buffers_.gpu_postprocess_frame_size * sizeof(float));
                    compute_binarise_otsu_bradley(buffers_.gpu_postprocess_frame,
                                                  d_output,
                                                  fd_.width,
                                                  fd_.height,
                                                  setting<settings::OtsuWindowSize>(),
                                                  setting<settings::OtsuLocalThreshold>(),
                                                  stream_);

                    cudaXMemcpy(buffers_.gpu_postprocess_frame,
                                d_output,
                                buffers_.gpu_postprocess_frame_size * sizeof(float),
                                cudaMemcpyDeviceToDevice);
                    cudaFree(d_output);
                }
                else
                    compute_binarise_otsu(buffers_.gpu_postprocess_frame, fd_.width, fd_.height, stream_);
            }
        });
}

} // namespace holovibes::compute
