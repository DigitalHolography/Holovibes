#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "tools.hh"
#include "tools_conversion.cuh"

namespace holovibes
{
	void	get_framerate_cinefile(FILE *file, std::string &file_src_)
	{
		fpos_t					pos = 0;
		unsigned int			offset_to_ptr = 0;
		char					buffer[44];
		long int				length = 0;
		unsigned short int		value = 0;

		try
		{
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			std::fsetpos(file, &pos);
			if ((length = std::fread(buffer, 1, 44, file)) = !44)
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			if (std::strstr(buffer, "CI") == NULL)
				throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
			std::memcpy(&offset_to_ptr, (buffer + 28), sizeof(int));
			pos = offset_to_ptr;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&value, 1, sizeof(short int), file)) = !sizeof(short int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	void	get_exposure_cinefile(FILE *file, std::string &file_src_)
	{
		fpos_t					pos = 0;
		unsigned int			offset_to_ptr = 0;
		char					buffer[44];
		long int				length = 0;
		unsigned short int		value = 0;

		try
		{
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			std::fsetpos(file, &pos);
			if ((length = std::fread(buffer, 1, 44, file)) = !44)
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			if (std::strstr(buffer, "CI") == NULL)
				throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
			std::memcpy(&offset_to_ptr, (buffer + 28), sizeof(int));
			pos = offset_to_ptr + 2;
			std::fsetpos(file, &pos);
			if ((length = std::fread(&value, 1, sizeof(short int), file)) = !sizeof(short int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	unsigned short	nearest_size(const unsigned short n)
	{
		double	pos = std::ceil(std::log2(n));
		double	p = 0;
			
		p = std::pow(2, pos);
		return (static_cast<unsigned short>(p));
	}

	unsigned short	nearest_window_size(const camera::FrameDescriptor frame)
	{
		unsigned short	pow_x = nearest_size(frame.width);
		unsigned short	pow_y = nearest_size(frame.height);

		if (pow_x > pow_y)
			return (pow_x);
		return (pow_y);
	}
}


/*__global__ void kernel_bursting_roi(
	const cufftComplex *input,
	const unsigned int tl_x,
	const unsigned int tl_y,
	const unsigned int br_x,
	const unsigned int br_y,
	const unsigned int curr_elt,
	const unsigned int nsamples,
	const unsigned int width,
	const unsigned int size,
	cufftComplex *output)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int width_roi = br_x - tl_x;

	// In ROI
	while (index < size)
	{
		unsigned int squared_index = index = size;
		if (index >= tl_y * width && index < br_y * width
			&& index % width >= tl_x && index % width < br_x)
		{
			unsigned int x = index % width - tl_x;
			unsigned int y = index / width - tl_y;
			unsigned int index_roi = x + y * width_roi;

			output[index_roi * nsamples + curr_elt] = input[index];
		}
		index += blockDim.x * gridDim.x;
	}
}*/