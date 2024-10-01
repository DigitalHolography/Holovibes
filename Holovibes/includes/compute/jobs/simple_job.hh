/*! \file
 *
 * \brief Implementation of the SimpleJob class
 */
#pragma once

#include "jobs/j  ob.hh"
#include "unique_ptr.hh"

namespace holovibes
{

/*! \brief class for all jobs which truly do some calculations
 */

class SimpleJob : public Job
{
  public:
    struct ExecuteEnv : Job::RunEnv
    {
        std::byte* output;
        Job::BufferDescriptor output_desc;

        explicit ExecuteEnv(const Job::RunEnv& env, std::byte* _output, const Job::BufferDescriptor& _output_desc)
            : Job::RunEnv(env)
            , output(_output)
            , output_desc(_output_desc)
        {
        }
    };

    explicit SimpleJob(shared_job next_job, bool inplace = false)
        : inplace_(inplace)
        , next_job_(next_job)
    {
    }

    SimpleJob(bool inplace = false)
        : inplace_(inplace)
    {
    }

    ~SimpleJob() override {}

    const std::byte* get_output_buffer() const noexcept { return output_.get(); }

    bool is_inplace() const noexcept { return inplace_; }

    shared_job get_next_job() const { return next_job_; }
    void set_next_job(shared_job job) { next_job_ = job; }

    void prepare(Job::BufferDescriptor input) final
    {
        check_input_dimensions(input);
        auto output = get_output_dimensions(input);
        if (inplace_ && output.get_buffer_size() > input.get_buffer_size())
            throw DimensionException{"An inplace job cannot have a bigger buffer output than input (input: "};

        if (!inplace_)
            output_.resize(output.get_buffer_size());

        if (next_job_ != nullptr)
            next_job_->prepare(output);
    }

    void run(Job::RunEnv env) final
    {
        auto new_env = ExecuteEnv{env, (inplace_) ? env.input : output_.get(), get_output_dimensions(env.input_desc)};
        execute(new_env);

        Job::RunEnv next_env = Job::RunEnv{env, new_env.output, new_env.output_desc};

        next_job_->run(next_env);
    }

    operator std::string() const override
    {
        return std::string{"SimpleJob{inplace = "} + std::to_string(inplace_) + " -> " + std::string{*next_job_};
    }

  protected:
    /*! \brief Check if the previous output frame size if valid for the job
     *
     * \param input The BufferDescriptor returned by the previous job in the list
     * \throw DimensionException when the buffer of input is unusable (ex: the job can only take square images)
     */
    virtual void check_input_dimensions(Job::BufferDescriptor input) = 0;

    /*! \brief Modify the current buffer descriptor with what will happen to the dimensions during the job
     *
     * \param input The BufferDescriptor returned by the previous job in the list
     * \return the new buffer descriptor
     */
    virtual Job::BufferDescriptor get_output_dimensions(Job::BufferDescriptor input) = 0;

    /*!
     * \brief The main function where everything is frozen except the frames
     *
     * \param input pointer on the input frame
     * \param desc the size and how to use the input buffer
     * \param stream the cuda stream to execute calculations
     */
    virtual void execute(ExecuteEnv env) = 0;

  private:
    /*! \brief Inplace means the Job's action is done on the buffer. (only side effect, no return) */
    const bool inplace_;
    shared_job next_job_{nullptr};
    cuda_tools::CudaUniquePtr<std::byte> output_{nullptr};
};

} // namespace holovibes
