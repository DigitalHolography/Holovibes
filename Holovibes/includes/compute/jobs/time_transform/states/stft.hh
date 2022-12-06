#pragma once

#include <cuComplex.h>
#include <cufft.h>

#include "cufft_handle.hh"
#include "jobs/time_transform/time_transform_state.hh"
#include "jobs/vector_job.hh"
#include "stft.cuh"
#include "jobs/simple_job.hh"

namespace holovibes
{
class TimeTransformState_STFT : public TimeTransformState
{
  public:
    TimeTransformState_STFT(TimeTransformJob* context, TimeTransformJob::TimeTransformEnv env)
        : TimeTransformState(context)
        , stft_plan_(env.stft_plan)
    {
    }

    void prepare(BufferDescriptor input) final {}

    void run(SimpleJob::ExecuteEnv env)
    {
        stft(reinterpret_cast<cuComplex*>(env.input), reinterpret_cast<cuComplex*>(env.output), stft_plan_);
    }

    

  private:
    cufftHandle stft_plan_;
};
} // namespace holovibes