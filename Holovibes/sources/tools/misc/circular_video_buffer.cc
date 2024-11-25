#include "circular_video_buffer.hh"
#include "cuda_memory.cuh"
#include "moments_treatments.cuh"
#include "tools_analysis.cuh"
#include "barycentre.cuh"

namespace holovibes
{
CircularVideoBuffer::CircularVideoBuffer(const size_t frame_res, const size_t buffer_capacity, cudaStream_t stream)
{
    // Allocate the internal GPU memory buffer
    data_.resize(buffer_capacity * frame_res);
    // Allocate the sum image buffer
    sum_image_.resize(frame_res);
    cudaXMemset(sum_image_, 0, frame_res * sizeof(float));

    start_index_ = 0;
    end_index_ = 0;
    buffer_capacity_ = buffer_capacity;
    nb_frames_ = 0;
    frame_res_ = frame_res;
    frame_size_ = frame_res * sizeof(float);
    stream_ = stream;
}

CircularVideoBuffer::~CircularVideoBuffer()
{
    data_.reset();
    sum_image_.reset();
    mean_image_.reset();
}

CircularVideoBuffer::CircularVideoBuffer(CircularVideoBuffer& ref)
{
    // Allocate the internal GPU memory buffer
    data_.resize(ref.buffer_capacity_ * ref.frame_res_);

    // Allocate the sum image buffer
    sum_image_.resize(ref.frame_res_);
    cudaXMemset(sum_image_, 0, ref.frame_res_ * sizeof(float));

    cudaXMemcpyAsync(data_.get(),
                     ref.data_.get(),
                     ref.frame_size_ * ref.nb_frames_,
                     cudaMemcpyDeviceToDevice,
                     ref.stream_);

    start_index_ = ref.start_index_;
    end_index_ = ref.end_index_;
    buffer_capacity_ = ref.buffer_capacity_;
    nb_frames_ = ref.nb_frames_;
    frame_res_ = ref.frame_res_;
    frame_size_ = ref.frame_size_;
    stream_ = ref.stream_;
}

CircularVideoBuffer& CircularVideoBuffer::operator=(CircularVideoBuffer& ref)
{
    cudaXMemcpyAsync(this->data_, ref.data_, ref.frame_size_ * ref.nb_frames_, cudaMemcpyDeviceToDevice, ref.stream_);

    cudaXMemcpyAsync(this->sum_image_,
                     ref.sum_image_,
                     ref.frame_size_ * ref.nb_frames_,
                     cudaMemcpyDeviceToDevice,
                     ref.stream_);

    this->start_index_ = ref.start_index_;
    this->end_index_ = ref.end_index_;
    this->buffer_capacity_ = ref.buffer_capacity_;
    this->nb_frames_ = ref.nb_frames_;
    this->frame_res_ = ref.frame_res_;
    this->frame_size_ = ref.frame_size_;
    this->stream_ = ref.stream_;
    return *this;
}

float* CircularVideoBuffer::get_first_frame()
{
    if (!nb_frames_)
        return nullptr;
    return data_ + start_index_ * frame_res_;
}

float* CircularVideoBuffer::get_last_frame()
{
    if (!nb_frames_)
        return nullptr;

    if (end_index_ == 0)
        return data_ + (buffer_capacity_ - 1) * frame_res_;

    return data_ + (end_index_ - 1) * frame_res_;
}

void CircularVideoBuffer::compute_mean_image()
{
    // Allocate mean_image_ buffer if first time called
    if (!mean_image_)
        mean_image_.resize(frame_size_);
    compute_mean(mean_image_, sum_image_, nb_frames_, frame_res_, stream_);
}

float* CircularVideoBuffer::get_mean_image() { return mean_image_.get(); }

void CircularVideoBuffer::add_new_frame(const float* const new_frame)
{
    float* new_frame_position = data_.get() + end_index_ * frame_res_;
    end_index_ = (end_index_ + 1) % buffer_capacity_;

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

size_t CircularVideoBuffer::get_frame_count() { return nb_frames_; }

float* CircularVideoBuffer::get_data_ptr() { return data_.get(); }

void CircularVideoBuffer::multiply_data_by_frame(float* frame)
{
    if (!compute_mean_1_2_buffer_)
        compute_mean_1_2_buffer_.resize(buffer_capacity_);
    cudaXMemsetAsync(compute_mean_1_2_buffer_, 0, sizeof(float) * nb_frames_, stream_);
    compute_multiplication_mean(compute_mean_1_2_buffer_, data_.get(), frame, frame_res_, nb_frames_, stream_);
}

float* CircularVideoBuffer::get_mean_1_2_() { return compute_mean_1_2_buffer_.get(); }

} // namespace holovibes