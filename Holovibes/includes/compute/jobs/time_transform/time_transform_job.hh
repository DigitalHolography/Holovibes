#pragma once

#include "jobs/time_transform/time_transform_state.hh"
#include "jobs/time_transform/states/stft.hh"
#include "jobs/time_transform/states/pca.hh"
#include "jobs/time_transform/states/ssa_stft.hh"
#include "enum_time_transformation.hh"
#include "jobs/job.hh"

namespace holovibes
{
class TimeTransformJob : public Job
{
  public:
    struct TimeTransformEnv
    {
        cuda_tools::CufftHandle stft_plan;
    };

  public:
    TimeTransformJob(shared_job next_job, bool inplace = false, TimeTransformState& state)
        : SimpleJob(next_job, inplace)
        , state_(state)
    {
    }

    TimeTransformJob(shared_job next_job, bool inplace = false, TimeTransformation state, TimeTransformEnv env)
        : SimpleJob(next_job, inplace)
    {
        change_state(state, env);
    }

    TimeTransformJob(bool inplace = false, TimeTransformState& state)
        : SimpleJob(inplace)
        , state_(state)
    {
    }

    TimeTransformJob(bool inplace = false, TimeTransformation state, TimeTransformEnv env)
        : SimpleJob(inplace)
    {
        change_state(state, env);
    }

    void prepare(Job::BufferDescriptor input) final { state_.prepare(input); }

    void run(Job::RunEnv env) final { state_.run(env); }

    /**
     * \brief change the state of the job (stft, pca, ssa_stft)
     * \param TimeTransformState state
     */
    void change_state(TimeTransformState& state) { state_ = state; }

    /**
     * \brief change the state of the job (stft, pca, ssa_stft)
     * \param TimeTransformation state
     * \param TimeTransformEnv env
     */
    void change_state(TimeTransformation state, TimeTransformEnv env)
    {
        switch (state)
        {
        case STFT:
            state_ = TimeTransformState_STFT(this, env);
            break;
        case STFT:
            state_ = TimeTransformState_PCA(this, env);
            break;
        case STFT:
            state_ = TimeTransformState_SSA_STFT(this, env);
            break;
        }
    }

  private:
    TimeTransformState& state_;
};
} // namespace holovibes
