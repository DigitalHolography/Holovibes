/*! \file
 *
 * \brief Implementation of the SimpleJob class
 */
#pragma once

#include "jobs/job.hh"
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

    explicit SimpleJob(json env, shared_job next_job, bool inplace = false)
        : inplace_(inplace)
        , env_(env)
        , next_job_(next_job)
    {
    }

    SimpleJob(json env, bool inplace = false)
        : inplace_(inplace)
        , env_(env)
    {
    }

    ~SimpleJob() override {}

    const std::byte* get_output_buffer() const noexcept { return output_.get(); }

    json get_env() const { return env_; }
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
        next_job_->prepare(output);
    }

    void run(Job::RunEnv env) final
    {
        auto new_env = ExecuteEnv{env, (inplace_) ? env.input : output_.get(), get_output_dimensions(env.input_desc)};
        execute(new_env);
        next_job_->run(new_env);
    }

    operator std::string() const override
    {
        return std::string{"SimpleJob{inplace = "} + std::to_string(inplace_) + ", env = " + std::string{env_} +
               "} -> " + std::string{*next_job_};
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
    const json env_;
    shared_job next_job_{nullptr};
    cuda_tools::UniquePtr<std::byte> output_{nullptr};
};

} // namespace holovibes