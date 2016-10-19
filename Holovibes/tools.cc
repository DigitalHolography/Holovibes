#include <stdexcept>
#include <iostream>
#include "tools.hh"

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
}