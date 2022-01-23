/*! \file
 *
 * \brief Implementation of the Job class
 */
#pragma once

#include <nlohmann/json.hpp>
#include "tools.hh"
#include "dimension_exception.hh"

namespace holovibes
{

/*! \brief meta-class for all jobs
 */
class Job
{
  public:
    struct BufferDescriptor
    {
        size_t get_frame_res() const noexcept { return width * height; }

        size_t get_frame_size() const noexcept { return depth * get_frame_res(); }

        size_t get_buffer_size() const noexcept { return nb_frames * get_frame_size(); }

        bool operator==(const BufferDescriptor& other) const noexcept
        {
            return width == other.width && height == other.height && depth == other.depth &&
                   nb_frames == other.nb_frames;
        }

        bool operator!=(const BufferDescriptor& other) const noexcept { return !(*this == other); }

        unsigned short width;
        unsigned short height;

        /*!
         * \brief number of bytes to represent 1 pixel
         *
         */
        unsigned short depth;
        unsigned short nb_frames;
    };

    struct RunEnv
    {
        std::byte* input;
        BufferDescriptor input_desc;

        const cudaStream_t& stream;

        RunEnv(std::byte* _input, BufferDescriptor _input_desc, const cudaStream_t& _stream)
            : input(_input)
            , input_desc(_input_desc)
            , stream(_stream)
        {
        }

        RunEnv(const RunEnv& env, std::byte* _input, BufferDescriptor _input_desc)
            : input(_input)
            , input_desc(_input_desc)
            , stream(env.stream)
        {
        }
    };

    virtual ~Job() {}

    /*!
     * \brief the prepare function is where we froze the format of the future frames to be passed in run()
     *        Note: it could be called multiple times in the life time of the object
     *
     * \param input input size to be passed in to the job
     * \throw DimensionException when the buffer of input is unusable (ex: the job can only take square images)
     */
    virtual void prepare(BufferDescriptor input) = 0;

    /*!
     * \brief The main function where everything is frozen except the frames
     *
     * \param buffers the run environment
     */
    virtual void run(RunEnv env) = 0;

    /*! \brief Used to ease debug */
    virtual operator std::string() const { return "Job{}"; }
};

using shared_job = std::shared_ptr<Job>;

} // namespace holovibes