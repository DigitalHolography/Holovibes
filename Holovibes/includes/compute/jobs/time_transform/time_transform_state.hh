#pragma once

#include "jobs/job.hh"
#include "jobs/time_transform/time_transform_job.hh"

namespace holovibes
{
class TimeTransformState : public Job
{
  public:
    TimeTransformState(TimeTransformJob* context)
        : context_(context)
    {
    }

  private:
    TimeTransformJob* context_;
};
} // namespace holovibes