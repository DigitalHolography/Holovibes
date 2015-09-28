
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

		struct FrameDescriptor
		{
		public:
			camera::FrameDescriptor	desc;
			/*! Width of the image. != frame width */
			unsigned short         img_width;
			/*! Height of the image. != frame height */
			unsigned short         img_height;


			/*
				http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c
				http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
			*/
			inline bool isPowerOfTwo(unsigned int x) const
			{
				return ((x != 0) && ((x & (~x + 1)) == x));
			}

			inline unsigned int nextHightestPowerOf2(unsigned int x) const
			{
				x--;
				x |= x >> 1;
				x |= x >> 2;
				x |= x >> 4;
				x |= x >> 8;
				x |= x >> 16;
				x++;
				return (x);
			}

			void		compute_sqared_image(void)
			{
				unsigned short	biggestBorder = (desc.width > desc.height ? desc.width : desc.height);

				img_width = desc.width;
				img_height = desc.height;

				if (desc.width != desc.height)
					desc.width = desc.height = biggestBorder;

				if (!isPowerOfTwo(biggestBorder))
					desc.width = desc.height = static_cast<unsigned short>(nextHightestPowerOf2(biggestBorder));
			}

			FrameDescriptor(camera::FrameDescriptor d)
				: desc(d)
				, img_width(d.width)
				, img_height(d.height)
			{
				this->compute_sqared_image();
			}
		};

		ThreadReader(std::string file_src
			, holovibes::ThreadReader::FrameDescriptor frame_desc
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
		holovibes::ThreadReader::FrameDescriptor	desc_;
		unsigned int frameId_;
		unsigned int spanStart_;
		unsigned int spanEnd_;
		Queue& queue_;

		std::thread thread_;
	};
}
#endif