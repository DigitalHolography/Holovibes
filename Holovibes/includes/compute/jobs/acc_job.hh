/*! \file
 *
 * \brief Implementation of the AccJob class
 */
#pragma once

#include "jobs/job.hh"
#include "unique_ptr.hh"

namespace holovibes
{

/*!
 * \brief The AccJob class handles the accumulation for his next job.
 *
 */
class AccJob : public Job
{
    explicit AccJob(shared_job job, unsigned short nb_frames)
        : next_job_(job)
        , nb_frames_(nb_frames)
    {
    }

    void prepare(Job::BufferDescriptor input) override { buffer_.resize(input.get_buffer_size() * nb_frames_); }

    void run(Job::RunEnv env) override
    {
        cudaXMemcpyAsync(buffer_.get() + (curr_nb_frames_++ * env.input_desc.get_buffer_size()),
                         env.input,
                         env.input_desc.get_buffer_size(),
                         cudaMemcpyDeviceToDevice,
                         env.stream);

        if (curr_nb_frames_ == nb_frames_)
        {
            env.input = buffer_.get();
            env.input_desc.nb_frames *= nb_frames_;
            next_job_.run(env);
            curr_nb_frames_ = 0;
        }
    }

    operator std::string() const override
    {
        return std::string{"AccJob{nb_frames = "} + std::to_string(nb_frames_) +
               ", curr_nb_frames = " + std::to_string(curr_nb_frames_) + "}";
    }

  private:
    shared_job next_job_;
    cuda_tools::UniquePtr<std::byte> buffer_{nullptr};

    /*! \brief number of frames to accumulate */
    unsigned short nb_frames_;

    /*! \brief counter */
    unsigned short curr_nb_frames_ = 0;
};
} // namespace holovibes