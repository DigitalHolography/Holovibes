/*! \file
 *
 * \brief Implementation of the Job class
 */
#pragma once

#include <json.hpp>
#include "tools.hh"

namespace holovibes
{

/*! \brief meta-class for all jobs
 */
class Job
{
  public:
    struct BufferDescriptor
    {
        size_t get_frame_res() const { return width * height; }

        size_t get_frame_size() const { return depth * get_frame_res(); }

        size_t get_buffer_size() const { return nb_frames * get_frame_size(); }

        unsigned short nb_frames;
        unsigned short width;
        unsigned short height;
        unsigned short depth;
    };

    struct BuffersEnv
    {
        std::byte* input;
        BufferDescriptor input_desc;

        std::byte* output;
        BufferDescriptor output_desc;
    };

    Job(json env, bool inplace = false)
        : env_(env)
        : inplace_(inplace)
    {
    }

    virtual ~Job();

    json get_env() const { return env_; }
    bool is_inplace() const { return inplace_; }

    /*! \brief returns the number of frames of accumulation needed in input */
    virtual unsigned short get_nb_frames_input(unsigned short previous) { return previous; }

    /*! \brief Check if the previous output frame size if valid for the job
     *
     * \param input The BufferDescriptor returned by the previous job in the list
     * \throw DimensionException when the buffer of input is unusable (ex: the job can only take square images)
     */
    __declspec(noreturn) virtual void check_input_dimensions(BufferDescriptor input) = 0;

    /*! \brief Modify the current buffer descriptor with what will happen to the dimensions during the job
     *
     * \param input The BufferDescriptor returned by the previous job in the list
     * \return the new buffer descriptor
     */
    virtual BufferDescriptor get_output_dimensions(BufferDescriptor input) = 0;

    /*!
     * \brief The main function where everything is frozen except the frames
     *
     * \param buffers the input and output buffers
     */
    virtual void run(BuffersEnv buffers) = 0;

  private:
    const json env_;
    const bool inplace_; // Inplace means the Job's action is done on the buffer. (only side effect, no return)
}
} // namespace holovibes