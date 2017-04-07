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

#include "thread_compute.hh"
#include "pipe.hh"

namespace holovibes
{
	ThreadCompute::ThreadCompute(
		ComputeDescriptor& desc,
		Queue& input,
		Queue& output,
		const PipeType pipetype)
		: compute_desc_(desc)
		, input_(input)
		, output_(output)
		, pipetype_(pipetype)
		, pipe_(nullptr)
		, memory_cv_()
		, thread_(&ThreadCompute::thread_proc, this)
	{
	}

	ThreadCompute::~ThreadCompute()
	{
		pipe_->request_termination();

		if (thread_.joinable())
			thread_.join();
	}

	void ThreadCompute::thread_proc()
	{
		//if (pipetype_ == PipeType::PIPE)
		pipe_ = std::shared_ptr<ICompute>(new Pipe(input_, output_, compute_desc_));
		//else
		//	pipe_ = std::shared_ptr<ICompute>(new Pipeline(input_, output_, compute_desc_));
		//	PIPELINE dead code !!!

		memory_cv_.notify_one();

		pipe_->exec();
	}
}
