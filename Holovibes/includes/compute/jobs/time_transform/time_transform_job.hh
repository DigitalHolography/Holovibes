#pragma once

#include "jobs/time_transform/time_transform_state.hh"
#include "jobs/time_transform/states/stft.hh"
#include "jobs/time_transform/states/pca.hh"
#include "jobs/time_transform/states/ssa_stft.hh"
#include "enum_time_transformation.hh"
#include "jobs/job.hh"

namespace holovibes
{
class TimeTransformJob : public SimpleJob
{
  public:
    TimeTransformJob(json env, shared_job next_job, bool inplace = false, TimeTransformState& state)
        : SimpleJob(env, next_job, inplace)
        , state_(state)
    {
    }

    TimeTransformJob(json env, bool inplace = false, TimeTransformState& state)
        : SimpleJob(env, inplace)
        , state_(state)
    {
    }

    TimeTransformJob(json env, bool inplace = false, TimeTransformation state)
        : SimpleJob(env, inplace)
    {
        change_state(state);
    }

    virtual void execute(ExecuteEnv env) { state_.handle(env); }

    void change_state(TimeTransformState& state) { state_ = state; }

    void change_state(TimeTransformation state)
    {
        switch (state)
        {
        case STFT:
            state_ = TimeTransformState_STFT(this);
            break;
        case STFT:
            state_ = TimeTransformState_PCA(this);
            break;
        case STFT:
            state_ = TimeTransformState_SSA_STFT(this);
            break;

        default:
            /* TODO what to do with NONE ?
             * Check before and remove TimeTransformJob ?
             * Just copy to next buffer
             */
            break;
        }
    }

  private:
    TimeTransformState& state_;
};
} // namespace holovibes
