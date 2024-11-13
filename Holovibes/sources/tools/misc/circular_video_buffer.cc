#include "circular_video_buffer.hh"
#include "cuda_memory.cuh"
#include "moments_treatments.cuh"

namespace holovibes
{
CircularVideoBuffer::CircularVideoBuffer(const size_t frame_res, const size_t buffer_capacity, cudaStream_t stream)
    : start_index_(0)
    , last_frame_index_(0)
    , nb_frames_(0)
    , buffer_capacity_(buffer_capacity)
    , frame_res_(frame_res)
    , frame_size_(frame_res * sizeof(float))
    , stream_(stream)
{
    float *data, *sum;
    // Allocate the internal GPU memory buffer
    cudaXMalloc(&data, buffer_capacity * frame_res_ * sizeof(float));
    // Allocate the sum image buffer
    cudaXMalloc(&sum, frame_res * sizeof(float));

    data_.reset(data);
    sum_image_.reset(sum);
}

CircularVideoBuffer::~CircularVideoBuffer()
{
    data_.reset(nullptr);
    sum_image_.reset(nullptr);
    mean_image_.reset(nullptr);
}

float* CircularVideoBuffer::get_first_frame()
{
    if (!nb_frames_)
        return nullptr;
    return data_ + start_index_ * frame_size_;
}

float* CircularVideoBuffer::get_last_frame()
{
    if (!nb_frames_)
        return nullptr;
    return data_ + last_frame_index_ * frame_size_;
}

void CircularVideoBuffer::compute_mean_image()
{
    // Allocate if first time
    if (!mean_image_)
    {
        float* data;
        cudaXMalloc(&data, frame_size_);
        mean_image_.reset(data);
    }

    compute_mean(mean_image_, sum_image_, nb_frames_, frame_res_, stream_);
}

void CircularVideoBuffer::add_new_frame(const float* const new_frame)
{
    last_frame_index_ = (last_frame_index_ + 1) % buffer_capacity_;
    float* new_frame_position = data_.get() + last_frame_index_ * frame_res_;

    // Check if we need to remove the oldest frame to make room for new frame
    if (nb_frames_ == buffer_capacity_)
    {
        float* oldest_frame = data_.get() + start_index_ * frame_res_;
        // Oldest frame is no longer part of the buffer, remove it from sum and shift start_index
        subtract_frame_from_sum(oldest_frame, frame_res_, sum_image_, stream_);
        start_index_ = (start_index_ + 1) % buffer_capacity_;
    }

    // Add new frame to sum
    add_frame_to_sum(new_frame, frame_res_, sum_image_, stream_);

    cudaXMemcpyAsync(new_frame_position, new_frame, frame_size_, cudaMemcpyDeviceToDevice, stream_);

    if (nb_frames_ < buffer_capacity_)
        ++nb_frames_;
}

bool CircularVideoBuffer::is_full() { return nb_frames_ == buffer_capacity_; }

} // namespace holovibes