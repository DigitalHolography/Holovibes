
#ifndef THREAD_READER_HH
#define THREAD_READER_HH

# include <iostream>
# include <thread>
#include <string>

# include "queue.hh"
# include "ithread_input.hh"

namespace holovibes
{
	class ThreadReader : public IThreadInput
	{
	public:
		ThreadReader(std::string file_src
			, camera::FrameDescriptor frame_desc
			, bool loop
			, unsigned int fps
			, unsigned int spanStart
			, unsigned int spanEnd
			, Queue& input);
		virtual ~ThreadReader();

		void  thread_proc(void);

		std::string file_src_;
		bool loop_;
		unsigned int fps_;
		camera::FrameDescriptor frame_desc_;
		unsigned int frameId_;
		unsigned int spanStart_;
		unsigned int spanEnd_;
		Queue& queue_;

		std::thread thread_;
	};
}
#endif