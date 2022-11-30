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
}

void ComputeWorker::stop()
{
    Worker::stop();
    api::get_compute_pipe().request_termination();
}

void ComputeWorker::run() { api::get_compute_pipe().exec(); }
} // namespace holovibes::worker
