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
#include <Windows.h>
#include <filesystem>

#include "tools.hh"
#include "tools_conversion.cuh"
#include "power_of_two.hh"

namespace holovibes
{
	unsigned short	upper_window_size(ushort width, ushort height)
	{
		return std::max(width, height);
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

	std::string get_exe_path()
	{
		char path[MAX_PATH];
		HMODULE hmodule = GetModuleHandle(NULL);
		if (hmodule != NULL)
		{
			GetModuleFileName(hmodule, path, (sizeof(path)));
			return path;
		}
		else
		{
			return "";
		}
	}

	std::string get_exe_dir()
	{
		char path[MAX_PATH];
		HMODULE hmodule = GetModuleHandle(NULL);
		if (hmodule != NULL)
		{
			GetModuleFileName(hmodule, path, (sizeof(path)));
			std::filesystem::path p(path);
			return p.parent_path().string();
		}
		else
		{
			return "";
		}
	}

	QString create_absolute_qt_path(const std::string& relative_path)
	{
		std::filesystem::path dir(get_exe_dir());
		dir = dir / relative_path;
		return QString(dir.string().c_str());
	}

	std::string create_absolute_path(const std::string& relative_path)
	{
		std::filesystem::path dir(get_exe_dir());
		dir = dir / relative_path;
		return dir.string();
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
}
