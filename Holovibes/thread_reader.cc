# include "thread_reader.hh"
# include <fstream>
# include <Windows.h>

namespace holovibes
{
	ThreadReader::ThreadReader(std::string file_src
		, ThreadReader::FrameDescriptor frame_desc
		, bool loop
		, unsigned int fps
		, unsigned int spanStart
		, unsigned int spanEnd
		, Queue& input)
		: IThreadInput()
		, file_src_(file_src)
		, frame_desc_(frame_desc.desc)
		, desc_(frame_desc)
		, loop_(loop)
		, fps_(fps)
		, frameId_(0)
		, spanStart_(spanStart)
		, spanEnd_(spanEnd)
		, queue_(input)
		, thread_(&ThreadReader::thread_proc, this)
	{
	}

	void	ThreadReader::thread_proc()
	{
		std::ifstream	ifs(file_src_, std::istream::in | std::ifstream::binary);
		unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
		char* buffer = new char[frame_size];

		for (int i = 0; i < frame_size; ++i)
			buffer[i] = 0;

		try
		{
			while (!stop_requested_)
			{
				if (!ifs.is_open())
					throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
				if (ifs.good() && frameId_ < spanEnd_)
				{
					do {
						for (unsigned int y = 0; y < desc_.img_height; ++y)
							ifs.read(buffer + y * frame_desc_.width * frame_desc_.depth, desc_.img_width * frame_desc_.depth);
					} while (++frameId_ < spanStart_);

					queue_.enqueue(buffer, cudaMemcpyHostToDevice);
					Sleep(1000 / fps_);
				}
				else
				{
					if (loop_)
					{
						if (ifs.is_open())
							ifs.close();
						ifs.open(file_src_, std::istream::in | std::ifstream::binary);
						frameId_ = 0;
					}
				}
			}
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
		if (ifs.is_open())
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
