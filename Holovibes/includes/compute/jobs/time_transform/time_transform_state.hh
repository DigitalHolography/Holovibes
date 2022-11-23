#pragma once

#include "jobs/job.hh"
#include "jobs/simple_job.hh"
#include "jobs/time_transform/time_transform_job.hh"

namespace holovibes
{
class TimeTransformState
{
  public:
    TimeTransformState(TimeTransformJob* context)
        : context_(context)
    {
    }

    virtual void handle(SimpleJob::ExecuteEnv env) = 0;

  private:
    TimeTransformJob* context_;
};
} // namespace holovibes