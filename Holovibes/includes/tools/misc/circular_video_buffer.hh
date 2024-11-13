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
    CircularVideoBuffer(size_t frame_res, size_t frame_capacity, cudaStream_t stream);

    float* get_first_frame();

    float* get_last_frame();

    void add_new_frame(float* new_frame);

  private:
    /*! \brief Video of the last 'time_window_' frames */
    cuda_tools::UniquePtr<float> data_{nullptr};

    /*! \brief Index of the first image of the buffer */
    size_t start_index_{0};

    /*! \brief Index of the last image of the buffer */
    size_t last_index_{0};

    /*! \brief Number of frames currently stored */
    size_t nb_frames_{0};

    /*! \brief Max number of frames that the buffer can store */
    size_t buffer_capacity_{0};

    /*! \brief Size of one frame in pixels */
    size_t frame_res_{0};

    /*! \brief Image with each pixel value equal to the sum of each value at the same pixel in the buffer */
    cuda_tools::UniquePtr<float> sum_image_{nullptr};

    /*! \brief Cuda stream used for async computations */
    cudaStream_t stream_;
};
} // namespace holovibes
