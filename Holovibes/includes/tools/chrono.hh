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

#pragma once

#include <chrono>

class Chrono
{
public:
	Chrono()
	{
		start();
	}

	void start()
	{
		start_ = std::chrono::steady_clock::now();
	}

	void stop()
	{
		end_ = std::chrono::steady_clock::now();
	}

	size_t get_seconds()
	{
		return std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
	}

	size_t get_milliseconds()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
	}

	size_t get_microseconds()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
	}

	size_t get_nanoseconds()
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
	}

private:
	std::chrono::time_point<std::chrono::steady_clock> start_;
	std::chrono::time_point<std::chrono::steady_clock> end_;
};