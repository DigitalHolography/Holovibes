#include "compute_worker.hh"

#include "holovibes.hh"
#include "pipe.hh"
#include "API.hh"

#include "cublas_handle.hh"
#include "cufft_handle.hh"
#include "cusolver_handle.hh"

namespace holovibes::worker
{
ComputeWorker::ComputeWorker()
    : Worker()
    , stream_(Holovibes::instance().get_cuda_streams().compute_stream)
{
    cuda_tools::CublasHandle::set_stream(stream_);
    cuda_tools::CufftHandle::set_stream(stream_);
    cuda_tools::CusolverHandle::set_stream(stream_);

    Holovibes::instance().init_gpu_queues();
    Holovibes::instance().create_pipe();
    Holovibes::instance().sync_pipe();
}

void ComputeWorker::run()
{
    while (!stop_requested_)
    {
        api::get_compute_pipe().sync_and_refresh();
        if (!stop_requested_)
            api::get_compute_pipe().exec();
    }
    api::get_compute_pipe().sync_and_refresh();
    Holovibes::instance().destroy_pipe();
    Holovibes::instance().destroy_gpu_queues();
    LOG_TRACE("Compute worker finally stop");
}
} // namespace holovibes::worker
