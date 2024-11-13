/*!
    \file circular_video_buffer.hh
*/
#pragma once

#include "unique_ptr.hh"

namespace holovibes
{
/*!
 * \brief A circular buffer that stores a fixed number of video frames.
 *
 * This class implements a circular buffer that stores a fixed number of
 * video frames, where the oldest frame is overwritten by the newest frame
 * when the buffer is full.
 *
 * The class provides methods for adding new frames to the buffer,
 * accessing the oldest and newest frames, and checking the size.
 */
class CircularVideoBuffer
{
  public:
    CircularVideoBuffer(const size_t frame_res, const size_t frame_capacity, cudaStream_t stream);

    ~CircularVideoBuffer();

    float* get_first_frame();

    float* get_last_frame();

    void compute_mean_image();

    float* get_mean_image();

    void add_new_frame(const float* const new_frame);

    bool is_full();

    // TODO: function to subtract / add an image to each frames of the buffer (centered)

  private:
    /*! \brief Video of the last 'time_window_' frames */
    cuda_tools::UniquePtr<float> data_{};

    /*! \brief Index of the first image of the buffer */
    size_t start_index_;

    /*! \brief Index of the index AFTER the last image of the buffer */
    size_t end_index_;

    /*! \brief Number of frames currently stored */
    size_t nb_frames_;

    /*! \brief Max number of frames that the buffer can store */
    size_t buffer_capacity_;

    /*! \brief Resolution of one frame in pixels */
    size_t frame_res_;

    /*! \brief Size of one frame in bytes */
    size_t frame_size_;

    /*! \brief Image with each pixel value equal to the sum of each value at the same pixel in the buffer */
    cuda_tools::UniquePtr<float> sum_image_{};

    /*! \brief Image with each pixel value equal to the mean of each value at the same pixel in the buffer */
    cuda_tools::UniquePtr<float> mean_image_{};

    /*! \brief Cuda stream used for async computations */
    cudaStream_t stream_;
};
} // namespace holovibes
