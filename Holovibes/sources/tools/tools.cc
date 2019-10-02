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

#include <iomanip>
#include <sstream>

#include "tools.hh"
#include "tools_conversion.cuh"
#include "power_of_two.hh"

namespace holovibes
{
	void	get_framerate_cinefile(FILE *file, std::string &file_src_)
	{
		fpos_t					pos = 0;
		unsigned int			offset_to_ptr = 0;
		char					buffer[44];
		size_t					length = 0;
		unsigned short int		value = 0;

		try
		{
			/*Checking if it exists*/
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			std::fsetpos(file, &pos);
			/*Reading the whole cine file header*/
			//if ((length = std::fread(buffer, 1, 44, file)) != 44)
			if (std::fread(buffer, 1, 44, file) != 44)
				throw std::runtime_error("[READER] unable to read file: " + file_src_);

			/*Checking if the file is actually a .cine file*/
			if (std::strstr(buffer, "CI") == NULL)
				throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
			/*Reading OffSetup for offset to CAMERA SETUP*/
			std::memcpy(&offset_to_ptr, (buffer + 28), sizeof(int));
			pos = offset_to_ptr;
			/*Reading value FrameRate16*/
			std::fsetpos(file, &pos);
			//if ((length = std::fread(&value, 1, sizeof(short int), file)) != sizeof(short int))
			if (std::fread(&value, 1, sizeof(short int), file) != sizeof(short int))
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
		size_t					length = 0;
		unsigned short int		value = 0;

		try
		{
			/*Checking if it exists*/
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			std::fsetpos(file, &pos);
			/*Reading the whole cine file header*/
			//if ((length = std::fread(buffer, 1, 44, file)) != 44)
			if (std::fread(buffer, 1, 44, file) != 44)
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
			/*Checking if the file is actually a .cine file*/
			if (std::strstr(buffer, "CI") == NULL)
				throw std::runtime_error("[READER] file " + file_src_ + " is not a valid .cine file");
			/*Reading OffSetup for offset to CAMERA SETUP*/
			std::memcpy(&offset_to_ptr, (buffer + 28), sizeof(int));
			/*Reading value Shutter16*/
			pos = offset_to_ptr + 2;
			std::fsetpos(file, &pos);
			//if ((length = std::fread(&value, 1, sizeof(short int), file)) != sizeof(short int))
			if (std::fread(&value, 1, sizeof(short int), file) != sizeof(short int))
				throw std::runtime_error("[READER] unable to read file: " + file_src_);
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	unsigned short	upper_window_size(ushort width, ushort height)
	{
		return std::max(width, height);
	}

	void print_gpu_buffer(const float* buf, std::size_t nb_elts)
	{
		float* tmp_buf = (float *)malloc(nb_elts * sizeof(float));
		if (!tmp_buf)
			return;
		cudaMemcpy(tmp_buf, buf, nb_elts * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < nb_elts; i++)
			std::cout << "i = " << i << ", value = " << tmp_buf[i] << std::endl;
		free(tmp_buf);
	}

	void print_gpu_buffer(const double* buf, std::size_t nb_elts)
	{
		double* tmp_buf = (double *)malloc(nb_elts * sizeof(double));
		if (!tmp_buf)
			return;
		cudaMemcpy(tmp_buf, buf, nb_elts * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < nb_elts; i++)
			std::cout << "i = " << i << ", value = " << tmp_buf[i] << std::endl;
		free(tmp_buf);
	}

	void print_gpu_buffer(const cuComplex* buf, std::size_t nb_elts)
	{
		cuComplex* tmp_buf = (cuComplex *)malloc(nb_elts * sizeof(cuComplex));
		if (!tmp_buf)
			return;
		cudaMemcpy(tmp_buf, buf, nb_elts * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		for (int i = 0; i < nb_elts; i++)
			std::cout << "i = " << i << ", x = " << tmp_buf[i].x << ", y = " << tmp_buf[i].y << std::endl;
		free(tmp_buf);
	}


	void get_good_size(ushort& width, ushort& height, ushort window_size)
	{
		if (width > height)
		{
			height = window_size * height / width;
			width = window_size;
		}
		else
		{
			width = window_size * width / height;
			height = window_size;
		}
	}
}

std::string engineering_notation(double value, int nb_significant_figures)
{

	static std::string prefix[] = {
		"y", "z", "a", "f", "p", "n", "�", "m", "",
		"k", "M", "G", "T", "P", "E", "Z", "Y"
	};


	if (value == 0.)
	{
		return "0";
	}

	std::string res;

	if (value < 0)
	{
		res += '-';
		value = -value;
	}

	int expof10 = log10(value);
	if (expof10 > 0)
		expof10 = (expof10 / 3) * 3;
	else
		expof10 = (-expof10 + 3) / 3 * (-3);

	value *= pow(10, -expof10);

	std::string SI_prefix_symbol = prefix[expof10 / 3 + 8];

	int leading_figure = static_cast<int>(log10(value)) % 3 + 1;

	std::stringstream ss;
	ss << std::fixed << std::setprecision(std::max(nb_significant_figures - leading_figure, 0))
		<< value << " " << SI_prefix_symbol;

	return ss.str();

	/*if (value >= 1000.)
	{
		value /= 1000.0; expof10 += 3;
	}
	else if (value >= 100.0)
		digits -= 2;
	else if (value >= 10.0)
		digits -= 1;

	if (numeric || (expof10 < PREFIX_START) ||
		(expof10 > PREFIX_END))
		sprintf(res, "%.*fe%d", digits - 1, value, expof10);
	else
		sprintf(res, "%.*f %s", digits - 1, value,
			prefix[(expof10 - PREFIX_START) / 3]);
	return result;*/
}
