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

/*! \file
 *
 * Queue class is a custom circular FIFO data structure. It can handle
 * CPU or GPU data. This class is used to store the raw images, provided
 * by the camera, and holograms. */
#pragma once

# include <iostream>
# include <mutex>

# include <cuda_runtime.h>

# include "frame_desc.hh"
# include "unique_ptr.hh"
#include "ithread_input.hh"

namespace holovibes
{
	/*! \brief Queue class is a custom circular FIFO data structure. It can handle
	** CPU or GPU data. This class is used to store the raw images, provided
	** by the camera, and holograms.
	**
	** This Queue is thread safe, it is impossible to enqueue and dequeue simultaneously.
	** As well as it is impossible to get a value from the class getters while another
	** object is enqueuing or dequeuing.
	**
	** The Queue ensures that all elements it contains are written in little endian.
	*/
	class Queue
	{
	public:
		/*! \brief Queue constructor
		**
		** Please note that every size used in internal allocations for the Queue depends
		** on provided FrameDescriptor, i-e in frame_size() and frame_res() methods.
		**
		** Please note that when you allocate a Queue, its element number elts should be at least greater
		** by 2 that what you need (e-g: 10 elements Queue should be allocated with a elts of 12).
		**
		** \param frame_desc Either the FrameDescriptor of the camera that provides
		** images or a FrameDescriptor used for computations.
		** \param elts Max number of elements that the queue can contain.
		**/
		Queue(const camera::FrameDescriptor& frame_desc, const unsigned int elts, std::string name, unsigned int input_width = 0, unsigned int input_height = 0, unsigned int elm_size = 1);
		~Queue();

		/*! \return the size of one frame (i-e element) of the Queue in bytes. */
		size_t get_frame_size() const;

		/*! \brief Empty the Queue and change its size.
		**
		**  \param size the new size of the Queue
		*/
		void resize(const unsigned int size);

		/*! \return pointer to internal buffer that contains data. */
		void* get_buffer();

		/*! \return FrameDescriptor of the Queue */
		const camera::FrameDescriptor& get_frame_desc() const;

		/*! \return the size of one frame (i-e element) of the Queue in pixels. */
		int get_frame_res();

		/*! \return the cuda stream associated */
		cudaStream_t get_stream() const;

		/*! \return the number of elements the Queue currently contains.
		**  As this is the most used method, it is inlined here.
		*/
		size_t get_current_elts() const
		{
			return curr_elts_;
		}

		/*! \return the number of elements the Queue can contains at its maximum. */
		unsigned int get_max_elts() const;

		/*! \return pointer to first frame. */
		void* get_start();

		/*! \return index of first frame (as the Queue is circular, it is not always zero). */
		unsigned int get_start_index();

		/*! \return pointer right after last frame */
		void* get_end();

		/*! \return pointer to end_index - n frame */
		void* get_last_images(const unsigned n);

		/*! \return index of the frame right after the last one containing data */
		unsigned int get_end_index();

		/*! \brief Enqueue method
		**
		** Copies the given elt according to cuda_kind cuda memory type, then convert
		** to little endian if the camera is in big endian.
		**
		** If the maximum element number has been reached, the Queue overwrite the first frame.
		**
		** \param elt pointer to element to enqueue
		** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice ...)
		** \param mode Wether elt should be : copied as it is | copied into a bigger square | cropped into a smaller square
		*/
		bool enqueue(void* elt, cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice);

		/*! \brief Dequeue method overload
		**
		** Copy the first element of the Queue into dest according to cuda_kind
		** cuda memory type then update internal attributes.
		**
		** \param dest destination of element copy
		** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice ...)
		*/
		void dequeue(void* dest, cudaMemcpyKind cuda_kind);

		/*! \brief Dequeue method overload for composite recording
		**
		** Copy the first element of the Queue into dest according to cuda_kind
		** cuda memory type ignoring one byte over two, 
		** then update internal attributes.
		**
		** \param dest destination of element copy
		** \param cuda_kind kind of memory transfer (e-g: CudaMemCpyHostToDevice ...)
		*/
		void dequeue_48bit_to_24bit(void* dest, cudaMemcpyKind cuda_kind);

		/*! \brief Dequeue method
		**
		** Update internal attributes (reduce Queue current elements and change start pointer)
		*/
		void dequeue();

		/*! Empties the Queue. */
		void clear();

		/* allow us to choose if we want to display the queue or not */
		void set_display(bool value);

		void set_square_input_mode(SquareInputMode mode);

		/*Create a string containing the buffer size in MB*/
		std::string calculate_size(void) const;

		std::mutex&	getGuard();

	private:
		void display_queue_to_InfoManager() const;

		std::mutex				mutex_;
		std::string				name_;
		camera::FrameDescriptor	frame_desc_;
		const size_t			frame_size_;
		const int				frame_resolution_;
		unsigned int		    max_elts_;
		size_t					curr_elts_;
		unsigned int			start_index_;
		const bool				is_big_endian_;
		cuda_tools::UniquePtr<char>	data_buffer_;
		cudaStream_t			stream_;
		bool					display_;

		//utils used for square input mode
		//Original size of the input
		unsigned int input_width_;
		unsigned int input_height_;
		unsigned int elm_size_;
		SquareInputMode square_input_mode_;
	};
}