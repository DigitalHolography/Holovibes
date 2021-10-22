#include "compute_worker.hh"

#include "holovibes.hh"
#include "pipe.hh"

#include "cublas_handle.hh"
#include "cufft_handle.hh"
#include "cusolver_handle.hh"

namespace holovibes::worker
{
ComputeWorker::ComputeWorker(std::atomic<std::shared_ptr<Pipe>>& pipe,
                             std::atomic<std::shared_ptr<BatchInputQueue>>& input,
                             std::atomic<std::shared_ptr<Queue>>& output)
    : Worker()
    , pipe_(pipe)
    , input_(input)
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

    pipe_.load()->request_termination();
}

void ComputeWorker::run()
{
    pipe_.load()->exec();

    pipe_.store(nullptr);
    output_.store(nullptr);
}
} // namespace holovibes::worker