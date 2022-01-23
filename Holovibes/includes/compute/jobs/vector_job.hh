/*! \file
 *
 * \brief Implementation of the VectorJob class
 */
#pragma once

#include "jobs/job.hh"
#include "jobs/simple_job.hh"

namespace holovibes
{

/*!
 * \brief The VectorJob class handles one or more Job and/or VectorJob. It creates and allocate their buffers, and its
 * run() method runs all the Jobs.
 *
 */
class VectorJob : public Job
{
  public:
    using const_iterator = std::vector<shared_job>::const_iterator;

    VectorJob() = default;

    VectorJob(std::vector<shared_job> jobs)
        : jobs_(jobs)
    {
        for (size_t i = 1; i < jobs.size(); i++)
        {
            auto simple_job = dynamic_cast<SimpleJob*>(jobs.at(i - 1).get());
            if (simple_job)
                simple_job->set_next_job(jobs.at(i));
        }
    }

    shared_job at(size_t i) const noexcept { return jobs_.at(i); }
    bool empty() const noexcept { return jobs_.empty(); }
    size_t size() const noexcept { return jobs_.size(); }
    void clear() noexcept { return jobs_.clear(); }

    shared_job front() const { return jobs_.front(); }
    shared_job back() const { return jobs_.back(); }

    const_iterator begin() { return jobs_.begin(); }
    const_iterator end() { return jobs_.end(); }

    operator std::string() const override { return std::string{*front()}; }

    void push_back(shared_job job)
    {
        if (!empty())
        {
            auto simple_job = dynamic_cast<SimpleJob*>(jobs_.back().get());
            if (simple_job)
                simple_job->set_next_job(job);
        }

        jobs_.push_back(job);
    }

    template <typename JobType, typename... Args>
    std::shared_ptr<JobType> emplace_back(Args&&... args)
    {
        auto job = std::make_shared<JobType>(std::forward<Args>(args)...);
        push_back(job);
        return job;
    }

    void prepare(Job::BufferDescriptor input) override
    {
        if (!empty())
            front()->prepare(input);
    }

    void run(Job::RunEnv env) override
    {
        if (!empty())
            front()->run(env);
    }

  private:
    std::vector<shared_job> jobs_;
};

} // namespace holovibes