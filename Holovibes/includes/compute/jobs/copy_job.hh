#pragma once

#include "jobs/simple_job.hh"

namespace holovibes
{

/*! \brief Simple job copying input to output
 */
class CopyJob : public SimpleJob
{
  public:
    CopyJob()
        : SimpleJob(json{}, nullptr, false)
    {
    }

    CopyJob(shared_job next_job)
        : SimpleJob(json{}, next_job, false)
    {
    }

    operator std::string() const override { return "CopyJob{}"; }

  protected:
    void check_input_dimensions(Job::BufferDescriptor input) override {}

    Job::BufferDescriptor get_output_dimensions(Job::BufferDescriptor input) override { return input; }

    void execute(ExecuteEnv env) override
    {
        cudaXMemcpyAsync(env.output, env.input, env.input_desc.get_buffer_size(), cudaMemcpyDeviceToDevice, env.stream);
    }
};
} // namespace holovibes