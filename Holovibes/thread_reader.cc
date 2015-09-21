# include "thread_reader.hh"
# include <fstream>
# include <Windows.h>

namespace holovibes
{
	ThreadReader::ThreadReader(std::string file_src
		, camera::FrameDescriptor frame_desc
		, bool loop
		, unsigned int fps
		, Queue& input)
		: IThreadInput()
		, file_src_(file_src)
		, frame_desc_(frame_desc)
		, loop_(loop)
		, fps_(fps)
		, queue_(input)
		, thread_(&ThreadReader::thread_proc, this)
	{
	}

	void	ThreadReader::thread_proc()
	{
		std::ifstream	ifs(file_src_, std::istream::in | std::ifstream::binary);
		unsigned int frame_size = frame_desc_.width * frame_desc_.height;
		char* buffer = new char[frame_size];
		std::cout << "start read file " << file_src_ << std::endl;

		try
		{
			while (!stop_requested_)
			{
				if (!ifs.is_open())
					throw std::runtime_error("[LOADER] unable to read/open file: " + file_src_);
				if (ifs.good())
				{
					ifs.read(buffer, frame_size);
					queue_.enqueue(buffer, cudaMemcpyHostToDevice);
					Sleep(1000 / fps_);
				}
				else
				{
					if (loop_)
					{
						ifs.close();
						ifs.open(file_src_, std::istream::in);
					}
					//else
					//	stop_requested_ = true;
				}
			}
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
		std::cout << "stop read file " << file_src_ << std::endl;
		ifs.close();
		stop_requested_ = true;
		delete[] buffer;
	}

	ThreadReader::~ThreadReader()
	{
		stop_requested_ = true;
		if (thread_.joinable())
			thread_.join();
	}
}