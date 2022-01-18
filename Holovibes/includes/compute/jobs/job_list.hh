/*! \file
 *
 * \brief Implementation of the JobList class
 */
#pragma once

#include "jobs/job.hh"

#include <stdexcept>

namespace holovibes
{

/*!
 * \brief The JobList class handles one or more Job and/or JobList. It creates and allocate their buffers, and its run()
 * method runs all the Jobs.
 *
 */
class JobList : public Job
{
  public:
    using const_iterator = std::vector<Job>::const_iterator;

    JobList(std::vector<Job> jobs) // The JobList is inplace IFF it contains only inplace Jobs
        : Job(json{}, std::all_of(jobs.begin(), jobs.end(), [](Job job) -> bool { return job.is_inplace(); }))
        , jobs_(jobs)
    {
        if (jobs.size() == 0)
            throw std::length_error("a job list cannot be empty")
    }

    const_iterator begin() { return jobs_.begin(); }
    const_iterator end() { return jobs_.end(); }

    unsigned short get_nb_frames_input(unsigned short previous) override
    {
        return jobs_.front().get_nb_frames_input(previous);
    }

    bool check_input_dimensions(BufferDescriptor input) override { return jobs_.front().check_input_dimensions(input); }

    BufferDescriptor get_output_dimensions(BufferDescriptor input) override
    {
        // We cannot do image accumulation on the first image since we dont own the buffer
        jobs_.front().check_input_dimensions(input);
        auto desc = jobs_.front().get_output_dimensions(input);

        for (size_t i = 1; i < jobs_.size(); i++)
        {
            // Fake Image accumulation to go forward in the buffers
            desc.nb_frames = jobs_.at(i).get_nb_frames_input(desc.nb_frames);

            jobs_.at(i).check_input_dimensions(desc);
            desc = jobs_.at(i).get_output_dimensions(desc);
        }

        return desc;
    }

    void run(Job::BuffersEnv buffers) override;
    *void create_buffers(const Job::BuffersEnv& buffers);

  private:
    std::vector<Job> jobs_;
    std::vector<cuda_tools::UniquePtr<std::byte>> gpu_memory_;
    std::vector<Job::BuffersEnv> buffers_;
};

} // namespace holovibes