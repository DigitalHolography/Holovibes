#include "circular_video_buffer.hh"
#include "cuda_memory.cuh"
#include "moments_treatments.cuh"

namespace holovibes
{
CircularVideoBuffer::CircularVideoBuffer(size_t frame_res, size_t buffer_capacity, cudaStream_t stream)
    : start_index_(0)
    , last_index_(0)
    , nb_frames_(0)
    , buffer_capacity_(buffer_capacity)
    , frame_res_(frame_res)
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

float* CircularVideoBuffer::get_first_frame()
{
    if (!nb_frames_)
        return nullptr;
    return data_ + start_index_ * frame_res_ * sizeof(float);
}

float* CircularVideoBuffer::get_last_frame()
{
    if (!nb_frames_)
        return nullptr;
    return data_ + last_index_ * frame_res_ * sizeof(float);
}

void CircularVideoBuffer::add_new_frame(float* new_frame)
{
    float* buffer_position = data_.get() + last_index_ * frame_res_;

    if (nb_frames_ == buffer_capacity_)
    {
        float* oldest_frame = data_.get() + start_index_ * frame_res_;

        add_frame_to_sum(frame_res_, oldest_frame, sum_image_, stream_);
    }

    subtract_frame_from_sum(frame_res_, new_frame, sum_image_, stream_);

    cudaXMemcpyAsync(buffer_position, new_frame, frame_res_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);

    last_index_ = (last_index_ + 1) % buffer_capacity_;

    if (nb_frames_ < buffer_capacity_)
        nb_frames_++;
    else
        start_index_ = (start_index_ + 1) % buffer_capacity_;
}

} // namespace holovibes