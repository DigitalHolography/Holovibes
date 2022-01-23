/*! \file
 *
 * \brief Implementation of the VectorJob class
 */
#pragma once

#include "jobs/output_job.hh"

namespace holovibes
{

/*!
 * \brief The VectorJob class handles one or more Job and/or VectorJob. It creates and allocate their buffers, and its
 * run() method runs all the Jobs.
 *
 */
class BufferOutputJob : public OutputJob
{
  public:
    /*! \brief Constructor
     *
     *  \param output output buffer location
     *  \param nb_frames we are going to implement to signal there is a new frame in the buffer
     *  \param max_size the maximum size in bytes of the images to put in
     */
    BufferOutputJob(std::byte* output, size_t& nb_frames, size_t max_size)
        : output_(output)
        , nb_frames_(nb_frames)
        , max_size_(max_size)
    {
    }

    void prepare(Job::BufferDescriptor input) override
    {
        if (input.get_buffer_size() > max_size_)
            throw DimensionException{"Input is to large for output"};
    }

    /*!
     * \brief The main function where everything is frozen except the frames
     *
     * \param buffers the run environment
     */
    void run(Job::RunEnv env) override
    {
        cudaXMemcpyAsync(output_, env.input, env.input_desc.get_buffer_size(), cudaMemcpyDeviceToDevice, env.stream);
        nb_frames_++;
    }

    /*! \brief Used to ease debug */
    operator std::string() const override { return "BufferOutputJob{max_size = " + std::to_string(max_size_) + "; }"; }

  private:
    std::byte* output_;
    size_t& nb_frames_;
    size_t max_size_;
};
} // namespace holovibes