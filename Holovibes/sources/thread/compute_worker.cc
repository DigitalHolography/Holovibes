/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "compute_worker.hh"

#include "holovibes.hh"
#include "pipe.hh"

namespace holovibes::worker
{
ComputeWorker::ComputeWorker(std::atomic<std::shared_ptr<ICompute>>& pipe,
                             std::atomic<std::shared_ptr<Queue>>& input,
                             std::atomic<std::shared_ptr<Queue>>& output)
    : Worker()
    , pipe_(pipe)
    , input_(input)
    , output_(output)
{
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
        auto output_fd = input_.load()->get_fd();
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
                                           Holovibes::instance().get_cd()));
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