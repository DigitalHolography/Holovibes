/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
    {}

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

            output_.store(std::make_shared<Queue>(
				output_fd, global::global_config.output_queue_max_size, Queue::QueueType::OUTPUT_QUEUE));

			pipe_.store(std::make_shared<Pipe>(*input_.load(), *output_.load(), Holovibes::instance().get_cd()));
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
}