#include "compute_worker.hh"

#include "holovibes.hh"
#include "pipe.hh"

#include "cublas_handle.hh"
#include "cufft_handle.hh"
#include "cusolver_handle.hh"

namespace holovibes::worker
{

ComputeWorker::ComputeWorker(std::atomic<std::shared_ptr<Pipe>>& pipe, std::atomic<std::shared_ptr<Queue>>& output)
    : Worker()
    , pipe_(pipe)
    , output_(output)
    , stream_(Holovibes::instance().get_cuda_streams().compute_stream)
{
    cuda_tools::CublasHandle::set_stream(stream_);
    cuda_tools::CufftHandle::set_stream(stream_);
    cuda_tools::CusolverHandle::set_stream(stream_);
}

void ComputeWorker::stop()
{
    Worker::stop();

    pipe_.load()->set_requested(ICS::Termination, true);
}

void ComputeWorker::run()
{
    pipe_.load()->exec();

    pipe_.store(nullptr);
    output_.store(nullptr);
}
} // namespace holovibes::worker