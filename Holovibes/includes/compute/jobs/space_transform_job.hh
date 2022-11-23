#pragma once

#include "jobs/state/context.hh"
#include "jobs/state/state.hh"
#include "jobs/simple_job.hh"

namespace holovibes
{
class SpaceTransformJob : public SimpleJob, public Context
{
  public:
    struct FFT1 : State
    {
        virtual void run()
        {
            
        }
    }
};
} // namespace holovibes