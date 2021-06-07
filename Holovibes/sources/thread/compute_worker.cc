#include "compute_worker.hh"

#include "holovibes.hh"
#include "pipe.hh"

#include "cublas_handle.hh"
#include "cufft_handle.hh"
#include "cusolver_handle.hh"

namespace holovibes::worker
{
ComputeWorker::ComputeWorker(
    std::atomic<std::shared_ptr<ICompute>>& pipe,
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
    try
    {
        auto& cd = Holovibes::instance().get_cd();
        camera::FrameDescriptor output_fd = input_.load()->get_fd();
        if (cd.compute_mode == Computation::Hologram)
        {
            output_fd.depth = 2;
            if (cd.img_type == ImgType::Composite)
                output_fd.depth = 6;
        }

        output_.store(
            std::make_shared<Queue>(output_fd,
                                    global::global_config.output_queue_max_size,
                                    Queue::QueueType::OUTPUT_QUEUE));

        pipe_.store(std::make_shared<Pipe>(*input_.load(),
                                           *output_.load(),
                                           Holovibes::instance().get_cd(),
                                           stream_));
    }
    catch (std::exception& e)
    {
        LOG_ERROR(e.what());
        return;
    }

    pipe_.load()->exec();

    pipe_.store(nullptr);
    output_.store(nullptr);
}
} // namespace holovibes::worker